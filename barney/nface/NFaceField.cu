// SPDX-FileCopyrightText:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0

#include "barney/common/barney-common.h"
#include "barney/nface/NFaceField.h"
#include "barney/Context.h"
#include "barney/nface/NFaceCuBQLSampler.h"
#include "barney/volume/MCGrid.cuh"
#if RTC_DEVICE_CODE
# include "rtcore/ComputeInterface.h"
# include "rtcore/TraceInterface.h"
#endif

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(/*file*/NFaceMC,/*name*/NFaceMC,
                       /*geomtype device data */
                       MCVolumeAccel<NFaceCuBQLSampler>::DD,false,false);
  RTC_IMPORT_USER_GEOM(/*file*/NFaceMC,/*name*/NFaceMC_Iso,
                       /*geomtype device data */
                       MCIsoSurfaceAccel<NFaceCuBQLSampler>::DD,false,false);

  NFaceField::PLD *NFaceField::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  __rtc_global void nfaceRasterCells(rtc::ComputeInterface ci,
                                     NFaceField::DD mesh,
                                     MCGrid::DD grid)
  {
#if RTC_DEVICE_CODE
    const int cellIdx = ci.launchIndex().x;
    if (cellIdx >= mesh.numCells) return;

    /* polyhedral cells are compact (Voronoi); splatting their
       bounding box into the macro-cell grid is a fine approximation
       (this is the same generic path all non-tet umesh elements
       take) */
    const box4f eltBounds = mesh.cellBounds(cellIdx);
    rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
#endif
  }

  MCGrid::SP NFaceField::buildMCs()
  {
    MCGrid::SP mcGrid = std::make_shared<MCGrid>(devices);
    buildInitialMacroCells(*mcGrid);
    return mcGrid;
  }

  /*! build *initial* macro-cell grid (ie, the scalar field min/max
    ranges, but not yet the majorants) over this field */
  void NFaceField::buildInitialMacroCells(MCGrid &grid)
  {
    if (grid.built()) {
      // initial grid already built
      return;
    }
    assert(!worldBounds.empty());

    float maxWidth = reduce_max(worldBounds.size());
    int MC_GRID_SIZE
      = 200 + int(sqrtf(numCells/1000.f));
    vec3i dims = 1+vec3i(worldBounds.size() * ((MC_GRID_SIZE-1) / maxWidth));
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.nf: building initial macro cell grid of " << dims << " MCs"
              << OWL_TERMINAL_DEFAULT << std::endl;
    grid.resize(dims);

    grid.gridOrigin  = worldBounds.lower;
    grid.gridSpacing = worldBounds.size() * rcp(vec3f(dims));

    grid.clearCells();

    for (auto device : *devices)
      __rtc_launch(device->rtc,
                   nfaceRasterCells,
                   divRoundUp(numCells,128),128,
                   getDD(device),grid.getDD(device));
    for (auto device : *devices)
      device->sync();
  }

  bool NFaceField::setData(const std::string &member,
                           const std::shared_ptr<Data> &value)
  {
    if (ScalarField::setData(member,value))
      return true;

    if (member == "vertex.position") {
      vertices = value->as<PODData>();
      return true;
    }
    if (member == "vertex.data") {
      scalars = value->as<PODData>();
      scalarsArePerVertex = true;
      return true;
    }
    if (member == "cell.data") {
      scalars = value->as<PODData>();
      scalarsArePerVertex = false;
      return true;
    }
    if (member == "face.index") {
      faceVertices = value->as<PODData>();
      return true;
    }
    if (member == "face.begin") {
      faceBegin = value->as<PODData>();
      return true;
    }
    if (member == "cell.face") {
      cellFaces = value->as<PODData>();
      return true;
    }
    if (member == "cell.begin") {
      cellBegin = value->as<PODData>();
      return true;
    }

    return false;
  }

  bool NFaceField::set1i(const std::string &member, const int &value)
  {
    if (ScalarField::set1i(member,value))
      return true;
    if (member == "flipOrientation") {
      flipOrientation = (value != 0);
      return true;
    }
    return false;
  }

  bool NFaceField::set1f(const std::string &member, const float &value)
  {
    if (ScalarField::set1f(member,value))
      return true;
    if (member == "epsRel") {
      epsRel = value;
      return true;
    }
    return false;
  }

#if RTC_DEVICE_CODE
  /*! cuBQL BVH build (refit_init) assumes finite, non-inverted prim
      boxes. Invalid connectivity (e.g. OOB indices) yields empty /
      ±inf bounds from cellBounds — sanitize to a tiny finite box so
      the builder does not corrupt GPU memory; sampling still rejects
      bad cells in cellScalar. */
  inline __rtc_device bool nfaceFiniteF(float x)
  {
    return (x == x) && (fabsf(x) < 1e37f);
  }

  inline __rtc_device bool nfacePrimBoundsSafeForCuBQL(const box4f &eb)
  {
    const box3f pb = getBox(eb);
    const range1f rw = getRange(eb);
    if (pb.empty())
      return false;
    if (!nfaceFiniteF(pb.lower.x) || !nfaceFiniteF(pb.lower.y)
        || !nfaceFiniteF(pb.lower.z) || !nfaceFiniteF(pb.upper.x)
        || !nfaceFiniteF(pb.upper.y) || !nfaceFiniteF(pb.upper.z))
      return false;
    if (!nfaceFiniteF(rw.lower) || !nfaceFiniteF(rw.upper))
      return false;
    if (rw.lower > rw.upper)
      return false;
    return true;
  }

  inline __rtc_device
  box4f nfaceSanitizePrimBoundsForCuBQL(const NFaceField::DD &mesh,
                                        const box4f &eb)
  {
    if (nfacePrimBoundsSafeForCuBQL(eb))
      return eb;
    const box3f &wb = mesh.worldBounds;
    if (!wb.empty() && nfaceFiniteF(wb.lower.x) && nfaceFiniteF(wb.lower.y)
        && nfaceFiniteF(wb.lower.z) && nfaceFiniteF(wb.upper.x)
        && nfaceFiniteF(wb.upper.y) && nfaceFiniteF(wb.upper.z)) {
      const vec3f mid = 0.5f * (wb.lower + wb.upper);
      const float pad = 1e-3f;
      const vec3f lo(mid.x - pad, mid.y - pad, mid.z - pad);
      const vec3f hi(mid.x + pad, mid.y + pad, mid.z + pad);
      return box4f(vec4f(lo, 0.f), vec4f(hi, 1.f));
    }
    return box4f(vec4f(0.f, 0.f, 0.f, 0.f), vec4f(1.f, 1.f, 1.f, 1.f));
  }
#endif

  __rtc_global
  void nfaceComputeElementBBs(rtc::ComputeInterface ci,
                              NFaceField::DD mesh,
                              box3f   *d_primBounds,
                              range1f *d_primRanges)
  {
#if RTC_DEVICE_CODE
    const int tid = ci.launchIndex().x;
    if (tid >= mesh.numCells) return;

    const box4f eb = nfaceSanitizePrimBoundsForCuBQL(mesh, mesh.cellBounds(tid));
    d_primBounds[tid] = getBox(eb);
    if (d_primRanges) d_primRanges[tid] = getRange(eb);
#endif
  }

  /*! computes, on specified device, the bounding boxes and - if
    d_primRanges is non-null - the primitive ranges. d_primBounds
    and d_primRanges (if non-null) must be pre-allocated and
    writeable on specified device */
  void NFaceField::computeElementBBs(Device  *device,
                                     box3f   *d_primBounds,
                                     range1f *d_primRanges)
  {
    __rtc_launch(device->rtc,nfaceComputeElementBBs,
                 divRoundUp(numCells,128),128,
                 getDD(device),d_primBounds,d_primRanges);
    device->rtc->sync();
  }

  NFaceField::NFaceField(Context *context,
                         const DevGroup::SP   &devices)
    : ScalarField(context,devices)
  {
    perLogical.resize(devices->numLogical);
  }

  __rtc_global
  void nfaceComputeWorldBounds(rtc::ComputeInterface ci,
                               NFaceField::DD mesh,
                               box3f *pWorldBounds)
  {
#if RTC_DEVICE_CODE
    int tid = ci.launchIndex().x;
    if (tid >= mesh.numCells)
      return;
    box4f bounds = mesh.cellBounds(tid);
    rtc::fatomicMin(&pWorldBounds->lower.x,bounds.lower.x);
    rtc::fatomicMin(&pWorldBounds->lower.y,bounds.lower.y);
    rtc::fatomicMin(&pWorldBounds->lower.z,bounds.lower.z);
    rtc::fatomicMax(&pWorldBounds->upper.x,bounds.upper.x);
    rtc::fatomicMax(&pWorldBounds->upper.y,bounds.upper.y);
    rtc::fatomicMax(&pWorldBounds->upper.z,bounds.upper.z);
#endif
  }

  /*! orientation diagnostic: computes each cell's signed volume via
      the divergence theorem (triangle fans over the faces, using the
      sign convention as configured); cells with negative volume are
      counted. If the majority of cells come out negative the sign
      convention is (very likely) flipped. */
  __rtc_global
  void nfaceCountFlippedCells(rtc::ComputeInterface ci,
                              NFaceField::DD mesh,
                              int *pNumFlipped)
  {
#if RTC_DEVICE_CODE
    const int cellIdx = ci.launchIndex().x;
    if (cellIdx >= mesh.numCells)
      return;
    const uint32_t cf0 = mesh.cellBegin[cellIdx];
    const uint32_t cf1 = mesh.cellBegin[cellIdx+1];
    if (cf0 >= cf1 || cf1 > mesh.numCellFaces)
      return;

    /* reference point for numerical robustness: first vertex of
       first face */
    bool haveOrigin = false;
    vec3f o(0.f,0.f,0.f);
    float vol6 = 0.f;
    for (uint32_t f = cf0; f < cf1; f++) {
      const int sf = mesh.cellFaces[f];
      const int faceIdx = (sf < 0 ? -sf : sf) - 1;
      if (faceIdx < 0 || faceIdx >= mesh.numFaces)
        return;
      const uint32_t fv0 = mesh.faceBegin[faceIdx];
      const uint32_t fv1 = mesh.faceBegin[faceIdx+1];
      if (fv1 < fv0+3 || fv1 > mesh.numFaceVertices)
        return;
      const float sign
        = (sf < 0) ? -mesh.orientationSign : mesh.orientationSign;
      const int ia = mesh.faceVertices[fv0];
      if (ia < 0 || ia >= mesh.numVertices)
        return;
      const vec3f a = mesh.vertices[ia];
      if (!haveOrigin) { o = a; haveOrigin = true; }
      for (uint32_t k = fv0+1; k+1 < fv1; k++) {
        const int ib = mesh.faceVertices[k];
        const int ic = mesh.faceVertices[k+1];
        if (ib < 0 || ib >= mesh.numVertices
            || ic < 0 || ic >= mesh.numVertices)
          return;
        const vec3f b = mesh.vertices[ib];
        const vec3f c = mesh.vertices[ic];
        vol6 += sign * dot(a-o,cross(b-o,c-o));
      }
    }
    if (vol6 < 0.f)
      ci.atomicAdd(pNumFlipped,1);
#endif
  }

  void NFaceField::commit()
  {
    if (!vertices)     throw std::runtime_error
      ("NFaceField: no 'vertex.position' set");
    if (!scalars)      throw std::runtime_error
      ("NFaceField: neither 'vertex.data' nor 'cell.data' set");
    if (!faceVertices) throw std::runtime_error
      ("NFaceField: no 'face.index' set");
    if (!faceBegin)    throw std::runtime_error
      ("NFaceField: no 'face.begin' set");
    if (!cellFaces)    throw std::runtime_error
      ("NFaceField: no 'cell.face' set");
    if (!cellBegin)    throw std::runtime_error
      ("NFaceField: no 'cell.begin' set");

    if (faceBegin->count < 2 || cellBegin->count < 2)
      throw std::runtime_error
        ("NFaceField: 'face.begin'/'cell.begin' must have numFaces+1/"
         "numCells+1 entries (>= 2)");
    /* the begin arrays are uint32, so the pools must be addressable
       with 32 bits */
    if (faceVertices->count > 0xffffffffull)
      throw std::runtime_error
        ("NFaceField: 'face.index' pool exceeds 2^32 entries - not "
         "representable with 32-bit 'face.begin' offsets");
    if (cellFaces->count > 0xffffffffull)
      throw std::runtime_error
        ("NFaceField: 'cell.face' pool exceeds 2^32 entries - not "
         "representable with 32-bit 'cell.begin' offsets");
    if (faceBegin->count > 0x7fffffffull+1ull
        || cellBegin->count > 0x7fffffffull+1ull)
      throw std::runtime_error
        ("NFaceField: more than 2^31 faces or cells not supported");

    this->numFaces = (int)(faceBegin->count - 1);
    this->numCells = (int)(cellBegin->count - 1);

    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      SetActiveGPU forDuration(device);

      assert(numCells > 0);
      if (!pld->pWorldBounds) {
        pld->pWorldBounds
          = (box3f*)rtc->allocMem(sizeof(box3f));
      }
      box3f emptyBox;
      rtc->copy(pld->pWorldBounds,&emptyBox,sizeof(emptyBox));
      __rtc_launch(rtc,nfaceComputeWorldBounds,
                   divRoundUp(numCells,128),128,
                   getDD(device),pld->pWorldBounds);
    }

    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      device->sync();
      // in case of having multiple devices this will repeatedly
      // download the same value; that's ok.
      PLD *pld = getPLD(device);
      device->rtc->copy(&worldBounds,
                        pld->pWorldBounds,
                        sizeof(worldBounds));
      device->sync();
      assert(!worldBounds.empty());
    }

    /* orientation sanity check (on the first device only): if the
       majority of cells have negative volume under the configured
       sign convention, the data almost certainly uses the opposite
       convention. */
    {
      Device *device = (*devices)[0];
      SetActiveGPU forDuration(device);
      auto rtc = device->rtc;
      int *d_numFlipped = (int*)rtc->allocMem(sizeof(int));
      int zero = 0;
      rtc->copy(d_numFlipped,&zero,sizeof(int));
      __rtc_launch(rtc,nfaceCountFlippedCells,
                   divRoundUp(numCells,128),128,
                   getDD(device),d_numFlipped);
      device->sync();
      int numFlipped = 0;
      rtc->copy(&numFlipped,d_numFlipped,sizeof(int));
      device->sync();
      rtc->freeMem(d_numFlipped);
      if (numFlipped > numCells/2)
        std::cout << OWL_TERMINAL_RED
                  << "#bn.nf: WARNING: " << numFlipped << " of " << numCells
                  << " cells have negative volume under the current "
                  << "orientation convention (positive face ID = outward "
                  << "normal" << (flipOrientation?", flipped":"") << "). "
                  << "The volume will render empty; if your data uses the "
                  << "opposite convention, set 'flipOrientation'."
                  << OWL_TERMINAL_DEFAULT << std::endl;
      else if (numFlipped > 0)
        std::cout << OWL_TERMINAL_RED
                  << "#bn.nf: WARNING: " << numFlipped << " of " << numCells
                  << " cells have negative volume - these cells have "
                  << "inconsistent face orientations (or are degenerate) "
                  << "and will not render correctly."
                  << OWL_TERMINAL_DEFAULT << std::endl;
    }
  }

  NFaceField::DD NFaceField::getDD(Device *device)
  {
    assert(device);
    NFaceField::DD dd;
    assert(vertices);
    assert(faceVertices);
    (ScalarField::DD &)dd = ScalarField::getDD(device);
    dd.vertices        = (const vec3f    *)vertices->getDD(device);
    dd.scalars         = (const float    *)scalars->getDD(device);
    dd.faceVertices    = (const int      *)faceVertices->getDD(device);
    dd.faceBegin       = (const uint32_t *)faceBegin->getDD(device);
    dd.cellFaces       = (const int      *)cellFaces->getDD(device);
    dd.cellBegin       = (const uint32_t *)cellBegin->getDD(device);
    dd.scalarsArePerVertex = scalarsArePerVertex;
    dd.numCells        = numCells;
    dd.numFaces        = numFaces;
    dd.numVertices     = (int)vertices->count;
    dd.numScalars      = (int)scalars->count;
    dd.numFaceVertices = (uint32_t)faceVertices->count;
    dd.numCellFaces    = (uint32_t)cellFaces->count;
    dd.orientationSign = flipOrientation ? -1.f : +1.f;
    dd.epsRel          = epsRel;
    assert(dd.numCells > 0);
    assert(dd.vertices);
    assert(dd.faceVertices);
    return dd;
  }

  IsoSurfaceAccel::SP NFaceField::createIsoAccel(IsoSurface *isoSurface)
  {
    auto sampler = std::make_shared<NFaceCuBQLSampler>(this);
    return std::make_shared<MCIsoSurfaceAccel<NFaceCuBQLSampler>>
      (isoSurface,
       createGeomType_NFaceMC_Iso,
       sampler);
  }

  VolumeAccel::SP NFaceField::createAccel(Volume *volume)
  {
    auto sampler
      = std::make_shared<NFaceCuBQLSampler>(this);
    return std::make_shared<MCVolumeAccel<NFaceCuBQLSampler>>
      (volume,
       createGeomType_NFaceMC,
       sampler);
  }
}
