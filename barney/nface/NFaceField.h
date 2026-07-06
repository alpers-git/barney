// SPDX-FileCopyrightText:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0

#pragma once

#include "barney/ModelSlot.h"
#include "barney/volume/MCAccelerator.h"

namespace BARNEY_NS {

  /*! scalar field over a polyhedral ("NFACE", CGNS NGON_n/NFACE_n
      style) unstructured mesh with *shared* faces:

      - every (polygonal) face is stored exactly once, as a list of
        0-based vertex indices in a flat pool (`faceVertices`), with a
        `faceBegin[numFaces+1]` offset array delimiting the faces;

      - every cell is a list of *signed, 1-based* face IDs in a flat
        pool (`cellFaces`), with a `cellBegin[numCells+1]` offset
        array. Following the CGNS SIDS convention, a *positive* face
        ID means the face's right-hand-rule normal points *out of*
        the cell, a negative ID means it points into the cell. (The
        `flipOrientation` parameter inverts this interpretation.)

      Compared to the VTK_POLYHEDRON face-stream path in UMeshField
      (which inlines every face's vertex list into each cell that
      uses it), this representation stores each interior face once,
      roughly halving face-connectivity memory - which is what makes
      very large polyhedral (e.g. Voronoi) meshes feasible without
      preprocessing.

      Cells are assumed CONVEX with (near-)planar faces (true for
      Voronoi cells by construction). Containment is a per-face
      plane test with early out, using Newell normals so mildly
      non-planar faces behave robustly. Non-convex cells may be
      classified incorrectly.

      Scalars can be per-cell ("cell.data") or per-vertex
      ("vertex.data"); per-vertex values are interpolated with
      inverse-distance weighting over the cell's face corners
      (matching the VTK_POLYHEDRON path in UMeshField). */
  struct NFaceField : public ScalarField
  {
    typedef std::shared_ptr<NFaceField> SP;

    NFaceField(Context *context,
               const DevGroup::SP &devices);

    virtual ~NFaceField() = default;

    /*! device-data for a shared-face polyhedral scalar field,
        containing all device-side pointers and functions to access
        this field and sample/evaluate its cells */
    struct DD : public ScalarField::DD {

      inline __rtc_device box4f cellBounds(uint32_t cellIdx) const;

      /* compute scalar of given cell at point P, and return that in
         'retVal'. returns true if P is inside the cell, false if
         outside (in which case retVal is not defined) */
      inline __rtc_device bool cellScalar(float &retVal,
                                          uint32_t cellIdx,
                                          vec3f P,
                                          bool dbg = false) const;

      const vec3f    *vertices;     //!< vertex positions
      const float    *scalars;      //!< per-vertex or per-cell scalars
      const int      *faceVertices; //!< flat pool of 0-based vertex indices
      const uint32_t *faceBegin;    //!< numFaces+1 offsets into faceVertices
      const int      *cellFaces;    //!< flat pool of signed 1-based face IDs
      const uint32_t *cellBegin;    //!< numCells+1 offsets into cellFaces
      int             numCells;
      int             numFaces;
      int             numVertices;
      int             numScalars;
      uint32_t        numFaceVertices; //!< size of faceVertices pool
      uint32_t        numCellFaces;    //!< size of cellFaces pool
      bool            scalarsArePerVertex;
      float           orientationSign; //!< +1, or -1 if flipOrientation
      float           epsRel;          //!< rel. containment tolerance
    };

    /*! build *initial* macro-cell grid (ie, the scalar field min/max
      ranges, but not yet the majorants) over this field */
    void buildInitialMacroCells(MCGrid &grid);

    /*! computes, on specified device, the bounding boxes and - if
      d_primRanges is non-null - the primitive ranges. d_primBounds
      and d_primRanges (if non-null) must be pre-allocated and
      writeable on specified device */
    void computeElementBBs(Device  *device,
                           box3f   *d_primBounds,
                           range1f *d_primRanges=0);

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setData(const std::string &member,
                 const std::shared_ptr<Data> &value) override;
    bool set1i(const std::string &member, const int &value) override;
    bool set1f(const std::string &member, const float &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    DD getDD(Device *device);

    /*! create, fill, and return a macrocell grid for this field */
    MCGrid::SP buildMCs() override;

    VolumeAccel::SP createAccel(Volume *volume) override;

    /*! creates an acceleration structure for a 'isoSurface' geometry
        using this scalar field type */
    IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;

    /*! returns part of the string used to find the optix device
        programs that operate on this type */
    static std::string typeName() { return "NFace"; };

    /*! @{ set by the user, as parameters */
    PODData::SP scalars;
    PODData::SP faceVertices;
    PODData::SP faceBegin;
    PODData::SP cellFaces;
    PODData::SP cellBegin;
    PODData::SP vertices;
    bool  scalarsArePerVertex = false;
    bool  flipOrientation     = false;
    float epsRel              = 1e-4f;
    /*! @} */
    int numCells = 0;
    int numFaces = 0;

    struct PLD {
      box3f *pWorldBounds = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
  };

  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================

  inline __rtc_device
  box4f NFaceField::DD::cellBounds(uint32_t cellIdx) const
  {
    const uint32_t cf0 = cellBegin[cellIdx];
    const uint32_t cf1 = cellBegin[cellIdx+1];
    if (cf0 >= cf1 || cf1 > numCellFaces)
      return box4f{};
    if (!scalarsArePerVertex && (int)cellIdx >= numScalars)
      return box4f{};

    box4f bb;
    for (uint32_t f = cf0; f < cf1; f++) {
      const int sf = cellFaces[f];
      const int faceIdx = (sf < 0 ? -sf : sf) - 1;
      if (faceIdx < 0 || faceIdx >= numFaces)
        return box4f{};
      const uint32_t fv0 = faceBegin[faceIdx];
      const uint32_t fv1 = faceBegin[faceIdx+1];
      if (fv1 < fv0+3 || fv1 > numFaceVertices)
        return box4f{};
      for (uint32_t k = fv0; k < fv1; k++) {
        const int vi = faceVertices[k];
        if (vi < 0 || vi >= numVertices)
          return box4f{};
        const int si = scalarsArePerVertex ? vi : (int)cellIdx;
        if (scalarsArePerVertex && si >= numScalars)
          return box4f{};
        bb.extend(vec4f(vertices[vi],scalars[si]));
      }
    }
    return bb;
  }

  inline __rtc_device
  bool NFaceField::DD::cellScalar(float &retVal,
                                  uint32_t cellIdx,
                                  vec3f P,
                                  bool dbg) const
  {
    const uint32_t cf0 = cellBegin[cellIdx];
    const uint32_t cf1 = cellBegin[cellIdx+1];
    if (cf0 >= cf1 || cf1 > numCellFaces)
      return false;

    /* ---- containment: P must be on the inside of every face's
       plane (cells are convex). Newell normal + centroid computed
       relative to the face's first vertex for numerical robustness
       with coordinates far from the origin. Both cells sharing a
       face evaluate the *identical* plane (only the sign differs),
       so this test is watertight across interior faces. ---- */
    for (uint32_t f = cf0; f < cf1; f++) {
      const int sf = cellFaces[f];
      const int faceIdx = (sf < 0 ? -sf : sf) - 1;
      if (faceIdx < 0 || faceIdx >= numFaces)
        return false;
      const uint32_t fv0 = faceBegin[faceIdx];
      const uint32_t fv1 = faceBegin[faceIdx+1];
      if (fv1 < fv0+3 || fv1 > numFaceVertices)
        return false;

      const int i0 = faceVertices[fv0];
      if (i0 < 0 || i0 >= numVertices)
        return false;
      const vec3f v0 = vertices[i0];
      vec3f N(0.f,0.f,0.f);
      vec3f C(0.f,0.f,0.f);
      vec3f prev(0.f,0.f,0.f); // = v0 - v0
      for (uint32_t k = fv0+1; k < fv1; k++) {
        const int vi = faceVertices[k];
        if (vi < 0 || vi >= numVertices)
          return false;
        const vec3f cur = vertices[vi] - v0;
        N = N + cross(prev,cur);
        C = C + cur;
        prev = cur;
      }
      /* (closing edge last->first contributes cross(prev,0) == 0) */
      const int n = int(fv1 - fv0);
      C = v0 + C * (1.f/n);

      const float sign = (sf < 0) ? -orientationSign : orientationSign;
      const vec3f PC = P - C;
      const float d = sign * dot(PC,N);
      /* d is |N| * (signed distance); the tolerance is scale
         invariant: allow P to be up to epsRel*|P-C| outside the
         plane. */
      const float eps = epsRel * sqrtf(dot(N,N)*dot(PC,PC)) + 1e-30f;
      if (d > eps)
        return false;
    }

    /* ---- scalar evaluation ---- */
    if (!scalarsArePerVertex) {
      if ((int)cellIdx >= numScalars)
        return false;
      retVal = scalars[cellIdx];
      return true;
    }

    /* per-vertex scalars: inverse-distance weighting over all face
       corners (vertices shared by k faces get counted k times; this
       matches UMeshField::polyScalar) */
    float weightSum = 0.f;
    float valueSum  = 0.f;
    for (uint32_t f = cf0; f < cf1; f++) {
      const int sf = cellFaces[f];
      const int faceIdx = (sf < 0 ? -sf : sf) - 1;
      const uint32_t fv0 = faceBegin[faceIdx];
      const uint32_t fv1 = faceBegin[faceIdx+1];
      for (uint32_t k = fv0; k < fv1; k++) {
        const int vi = faceVertices[k];
        if (vi < 0 || vi >= numScalars)
          return false;
        const vec3f vp = vertices[vi];
        const float dist2 = dot(vp-P,vp-P);
        const float w = 1.f/fmaxf(dist2,1e-20f);
        weightSum += w;
        valueSum  += w*scalars[vi];
      }
    }
    if (!(weightSum > 0.f))
      return false;
    retVal = valueSum/weightSum;
    return true;
  }

}
