// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/barney-common.h"
#include "barney/DeviceGroup.h"
#include "barney/volume/MCGrid.h"
#include "barney/volume/Volume.h"
#include "barney/geometry/IsoSurface.h"
#include "barney/volume/DDA.h"
#include "barney/render/World.h"
#include "barney/render/OptixGlobals.h"
#include "barney/material/DeviceMaterial.h"
#if RTC_DEVICE_CODE
# include "rtcore/TraceInterface.h"
#endif

namespace BARNEY_NS {
  using render::Ray;
  using render::DeviceMaterial;

#if RTC_DEVICE_CODE
  /*! Customization point for iso-surface shading normals: a field sampler may
      provide the reconstructed value *and* an analytic object-space gradient in
      a single query (e.g. the AMR basis method, ExaBricks eq. 3). Returning
      false - the default for samplers that don't specialise this - makes the iso
      accel fall back to central differences. ADL finds a sampler-specific
      overload defined in that sampler's own header. */
  template<typename SamplerDD>
  inline __rtc_device
  bool isoAnalyticGrad(const SamplerDD &, vec3f, float &, vec3f &) { return false; }

  template<typename SamplerDD>
  inline __rtc_device
  float isoLocalStep(const SamplerDD &, vec3f, float fallback) { return fallback; }
#endif

  template<typename SFSampler>
  struct MCVolumeAccel : public VolumeAccel
  {
    struct DD 
    {
      Volume::DD<SFSampler> volume;
      MajorantsGrid::DD     mcGrid;
    };

    struct PLD {
      rtc::Geom  *geom  = 0;
      rtc::Group *group = 0;
    };
    PLD *getPLD(Device *device) 
    { return &perLogical[device->contextRank()]; } 
    std::vector<PLD> perLogical;

    DD getDD(Device *device)
    {
      DD dd;
      dd.volume = volume->getDD(device,sfSampler);
      dd.mcGrid = majorantsGrid->getDD(device);
      return dd;
    }

    MCVolumeAccel(Volume *volume,
                  GeomTypeCreationFct creatorFct,
                  const std::shared_ptr<SFSampler> &sfSampler);
    ~MCVolumeAccel() override;

      GeomTypeCreationFct const creatorFct;
    
    void build(bool full_rebuild) override;

#if BARNEY_DEVICE_PROGRAM
    /*! optix bounds prog for this class of accels */
    static inline __rtc_device
    void boundsProg(const rtc::TraceInterface &ti,
                    const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    static inline __rtc_device
    void isProg(rtc::TraceInterface &ti);
#endif 
    
    MajorantsGrid::SP majorantsGrid;
    const std::shared_ptr<SFSampler> sfSampler;
  };
  
  template<typename SFSampler>
  struct MCIsoSurfaceAccel : public IsoSurfaceAccel 
  {
    struct DD 
    {
      IsoSurface::DD<SFSampler> isoSurface;
      MCGrid::DD mcGrid;
    };
    
    struct PLD {
      rtc::Geom  *geom  = 0;
      rtc::Group *group = 0;
    };
    PLD *getPLD(Device *device) 
    { return &perLogical[device->contextRank()]; } 
    std::vector<PLD> perLogical;
    
    DD getDD(Device *device)
    {
      DD dd;
      dd.isoSurface = isoSurface->getDD(device,sfSampler);
      // dd.sf     = sfSampler->getDD(device);
      dd.mcGrid = mcGrid->getDD(device);
      return dd;
    }
    
    MCIsoSurfaceAccel(IsoSurface *isoSurface,
                      GeomTypeCreationFct creatorFct,
                      const std::shared_ptr<SFSampler> &sfSampler);
    ~MCIsoSurfaceAccel() override;

    GeomTypeCreationFct const creatorFct;
    
    void build() override;
    
#if BARNEY_DEVICE_PROGRAM
    /*! optix bounds prog for this class of accels */
    static inline __rtc_device
    void boundsProg(const rtc::TraceInterface &ti,
                    const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    static inline __rtc_device
    void isProg(rtc::TraceInterface &ti);
#endif 
    
    MCGrid::SP       mcGrid;
    const std::shared_ptr<SFSampler> sfSampler;
  };
  
  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
  template<typename SFSampler>
  void MCVolumeAccel<SFSampler>::build(bool full_rebuild) 
  {
    if (!majorantsGrid) {
      auto mcGrid = volume->sf->getMCs();
      majorantsGrid = std::make_shared<MajorantsGrid>(mcGrid);
    }
    majorantsGrid->computeMajorants(&volume->xf);
    sfSampler->build();
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFct);
        // build a single-prim geometry, that single prim is our
        // entire MC/DDA grid
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
      }
      rtc::Geom *geom = pld->geom;
      DD dd = getDD(device);
      geom->setDD(&dd);
      
      if (!pld->group) {
        // now put that into a instantiable group, and build it.
        pld->group = device->rtc->createUserGeomsGroup({geom});
      }
      pld->group->buildAccel();
      
      // now let the actual volume we're building know about the
      // group we just created
      Volume::PLD *volumePLD = volume->getPLD(device);
      if (volumePLD->generatedGroups.empty()) 
        volumePLD->generatedGroups = { pld->group };
    }
  }



  template<typename SFSampler>
  void MCIsoSurfaceAccel<SFSampler>::build() 
  {
    mcGrid = isoSurface->sf->getMCs();
    sfSampler->build();
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFct);
        // build a single-prim geometry, that single prim is our
        // entire MC/DDA grid
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
      }
      rtc::Geom *geom = pld->geom;
      DD dd = getDD(device);
      geom->setDD(&dd);
      
      IsoSurface::PLD *isoSurfacePLD = isoSurface->getPLD(device);
      isoSurfacePLD->userGeoms = { geom };
      // if (!pld->group) {
      //   // now put that into a instantiable group, and build it.
      //   pld->group = device->rtc->createUserGeomsGroup({geom});
      // }
      // pld->group->buildAccel();

      // // now let the actual volume we're building know about the
      // // group we just created
      // IsoSurface::PLD *isoSurfacePLD = isoSurface->getPLD(device);
      // if (isoSurfacePLD->generatedGroups.empty()) 
      //   isoSurfacePLD->generatedGroups = { pld->group };
    }
  }
  
  

  template<typename SFSampler>
  MCVolumeAccel<SFSampler>::
  MCVolumeAccel(Volume *volume,
                GeomTypeCreationFct creatorFct,
                const std::shared_ptr<SFSampler> &sfSampler)
    : VolumeAccel(volume),
      sfSampler(sfSampler),
      creatorFct(creatorFct)
  {
    perLogical.resize(devices->numLogical);
  }

  template<typename SFSampler>
  MCVolumeAccel<SFSampler>::~MCVolumeAccel()
  {
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      if (pld->group) {
        device->rtc->freeGroup(pld->group);
        pld->group = 0;
      }
      if (pld->geom) {
        device->rtc->freeGeom(pld->geom);
        pld->geom = 0;
      }
    }
  }

  template<typename SFSampler>
  MCIsoSurfaceAccel<SFSampler>::
  MCIsoSurfaceAccel(IsoSurface *isoSurface,
                    GeomTypeCreationFct creatorFct,
                    const std::shared_ptr<SFSampler> &sfSampler)
    : IsoSurfaceAccel(isoSurface),
      sfSampler(sfSampler),
      creatorFct(creatorFct)
  {
    perLogical.resize(devices->numLogical);
  }

  template<typename SFSampler>
  MCIsoSurfaceAccel<SFSampler>::~MCIsoSurfaceAccel()
  {
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      if (pld->group) {
        device->rtc->freeGroup(pld->group);
        pld->group = 0;
      }
      // iw - do NOT free the geom we created - geometries free their
      // own geoms when they die, if we free here we'll get a double
      // free.
    }
  }
  
  // ------------------------------------------------------------------
  // device progs: macro-cell accel with DDA traversal
  // ------------------------------------------------------------------

#if BARNEY_DEVICE_PROGRAM && RTC_DEVICE_CODE
  template<typename SFSampler>
  inline __rtc_device
  void MCVolumeAccel<SFSampler>::boundsProg(const rtc::TraceInterface &ti,
                                            const void *geomData,
                                            owl::common::box3f &bounds,
                                            const int32_t primID)
  {
    const DD &self = *(DD*)geomData;
    bounds = self.volume.sfCommon.worldBounds;
  }
  
  template<typename SFSampler>
  inline __rtc_device
  void MCIsoSurfaceAccel<SFSampler>::boundsProg(const rtc::TraceInterface &ti,
                                                const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    const DD &self = *(DD*)geomData;
    bounds = self.isoSurface.sfCommon.worldBounds;
  }
  
  template<typename SFSampler>
  inline __rtc_device
  void MCIsoSurfaceAccel<SFSampler>::isProg(rtc::TraceInterface &ti)
  {
    const void *pd = ti.getProgramData();
           
    const DD &self = *(typename MCIsoSurfaceAccel<SFSampler>::DD*)pd;
    const render::World::DD &world = render::OptixGlobals::get(ti).world;
    // ray in world space
    Ray &ray = *(Ray*)ti.getPRD();
#ifdef NDEBUG
    const bool dbg = false;
#else
    const bool dbg = ray.dbg();
#endif
    
    box3f bounds = self.isoSurface.sfCommon.worldBounds;
    range1f tRange = { ti.getRayTmin(), ti.getRayTmax() };
    
    // ray in object space
    vec3f obj_org = ti.getObjectRayOrigin();
    vec3f obj_dir = ti.getObjectRayDirection();

    if (dbg) {
      printf("MCIsoAccel isec %f %f %f mcgrid %i %i %i\n",
             obj_dir.x,
             obj_dir.y,
             obj_dir.z,
             self.mcGrid.dims.x,
             self.mcGrid.dims.y,
             self.mcGrid.dims.z
             );
    }
    
    auto objRay = ray;
    objRay.org = obj_org;
    objRay.dir = obj_dir;

    if (!boxTest(objRay,tRange,bounds))
      return;

    // ------------------------------------------------------------------
    // Re-origin the fine march at the field-box entry so every sample
    // position below is formed from small, dataset-scale parameters rather
    // than camera-distance-scale ones. With a far camera P = obj_org +
    // t*obj_dir is a difference of two large, nearly-cancelling terms, so
    // consecutive samples along the ray carry independent ~|obj_org|*eps
    // position noise; once that approaches the finest cell size the crossing
    // search brackets inconsistently and the surface TEARS (grows as you zoom
    // out, clean up close). Shifting the origin to the box entry makes each
    // per-sample increment small and self-consistent - only a fixed, smoothly
    // varying sub-cell offset survives - so the march is zoom-invariant. Same
    // reason Spheres uses the RTG2 "move the origin" trick. All march-local t
    // below are measured from tEnter; convert back (+tEnter) only for the
    // reported hit distance / depth. Volume path is deliberately untouched.
    const float tEnter   = tRange.lower;               // box entry, original units
    const float tMaxL    = ray.tMax - tEnter;          // ray max, march-local
    const vec3f marchOrg = obj_org + tEnter * obj_dir; // per-ray base, near data
    // ------------------------------------------------------------------
    // compute ray in macro cell grid space
    // ------------------------------------------------------------------
    vec3f mcGridOrigin  = self.mcGrid.gridOrigin;
    vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    vec3f dda_org = marchOrg;
    vec3f dda_dir = obj_dir;

    dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    dda_dir = dda_dir * rcp(mcGridSpacing);

#if 1
    Random rng(ray.rngSeed,hash(ti.getRTCInstanceIndex(),
                                ti.getGeometryIndex(),
                                ti.getPrimitiveIndex()));
#else
    Random rng(ray.rngSeed.next(hash(ti.getRTCInstanceIndex(),
                                     ti.getGeometryIndex(),
                                     ti.getPrimitiveIndex())));
#endif
    
    // Object-space ray-direction length: converts a t-interval into a world
    // distance for the adaptive march step below (obj_dir is generally not
    // unit-length). refStep is only a fallback for samplers that don't provide a
    // local cell size (see the per-segment step in the DDA lambda).
    const float dirLen  = length(obj_dir);
    const float refStep = reduce_min(mcGridSpacing) * 0.1f;
    float tHit = tMaxL;
    dda::dda3(dda_org,dda_dir,tRange.upper - tEnter,
              vec3ui(self.mcGrid.dims),
              [&](const vec3i &cellIdx, float t0, float t1) -> bool
              {
                float _t0 = t0;
                float _t1 = t1;
                range1f tRange = range1f {t0,min(t1,tMaxL)};
                if (tRange.lower >= tRange.upper) return true;
                
                range1f valueRange = self.mcGrid.scalarRange(cellIdx);
                
                // scalar values at begin/end of current ray segment
                // (NOT sorted by value as valuerange is!)
                float ff0 = 0.f, ff1 = 0.f;
                if (dbg) printf("dda %i %i %i [%f %f] -> [%f %f]\n",
                                cellIdx.x,
                                cellIdx.y,
                                cellIdx.z,
                                tRange.lower,
                                tRange.upper,
                                valueRange.lower,
                                valueRange.upper);
                auto overlaps = [&](float isoValue)
                {
                  return
                    isoValue >= valueRange.lower &&
                    isoValue <= valueRange.upper;
                };
                if (!overlaps(self.isoSurface.isoValue))
                  return true;

                auto intersect = [&](float isoValue)
                {
                  // The basis reconstruction is not linear along the ray. The
                  // march only gives us a bracket, so refine that bracket before
                  // evaluating the shading gradient. Otherwise the reported hit
                  // can sit noticeably off the level set and its normal changes
                  // with the view direction.
                  float lo = tRange.lower, hi = tRange.upper;
                  float vlo = ff0, vhi = ff1;
                  for (int refine=0;refine<3;refine++) {
                    const float denom = vhi-vlo;
                    float u = denom == 0.f ? .5f : (isoValue-vlo)/denom;
                    u = min(.9f,max(.1f,u));
                    const float tm = lerp_l(u,lo,hi);
                    const float vm = self.isoSurface.sfSampler.sample
                      (marchOrg + tm*obj_dir,dbg);
                    if (isnan(vm)) break;
                    if ((vlo < isoValue) == (vm < isoValue)) {
                      lo = tm; vlo = vm;
                    } else {
                      hi = tm; vhi = vm;
                    }
                  }
                  const float denom = vhi-vlo;
                  const float u = denom == 0.f
                    ? .5f : min(1.f,max(0.f,(isoValue-vlo)/denom));
                  const float t = lerp_l(u,lo,hi);
                  tHit = min(tHit,t);
                };
                  

                float tt1 = t0;
                vec3f P = marchOrg + tt1 * obj_dir;
                ff1 = self.isoSurface.sfSampler.sample(P,dbg);
                // Anti-aliasing guard for the crossing search: sample several
                // times per FINEST cell the segment actually touches, or thin
                // fine-level crossings get stepped over and the surface drops out
                // for the views where the crossing projects thin (macro-cell-
                // aligned, view-dependent patches). Needed IN ADDITION to the
                // precision re-origin: re-origin makes the sample positions
                // accurate, this makes them dense enough to catch thin crossings;
                // removing either brings the tears back (verified). A macro cell
                // spans many fine AMR cells and can straddle a level boundary, so
                // probe the local cell size at both ends AND the middle of the
                // segment and take the finest (smallest) - a single midpoint probe
                // reports the coarse size when a fine block only touches one end,
                // marching the whole segment too coarse. isoLocalStep returns half
                // the local cell for AMR (fallback = macro-cell refStep for other
                // samplers); the extra 0.5 oversamples to ~4 samples per finest cell.
                const float sA = isoLocalStep(self.isoSurface.sfSampler,
                                              marchOrg + _t0*obj_dir, refStep);
                const float sB = isoLocalStep(self.isoSurface.sfSampler,
                                              marchOrg + (0.5f*(_t0+_t1))*obj_dir, refStep);
                const float sC = isoLocalStep(self.isoSurface.sfSampler,
                                              marchOrg + _t1*obj_dir, refStep);
                const float mStep = 0.5f * min(sA,min(sB,sC));
                int numSteps = 10;
                if (mStep > 0.f) {
                  const float segLen = (_t1 - _t0) * dirLen;
                  numSteps = int(ceilf(segLen / mStep));
                  numSteps = max(10, min(numSteps, 256));
                }
                for (int i=1;i<=numSteps;i++) {
                  float tt0 = tt1;
                  ff0 = ff1;
                  tt1 = lerp_l(i/float(numSteps),_t0,_t1);
                  P = marchOrg + tt1 * obj_dir;
                  ff1 = self.isoSurface.sfSampler.sample(P,dbg);
                  if (isnan(ff0) || isnan(ff1)) continue;
                  
                  valueRange.lower = min(ff0,ff1);
                  valueRange.upper = max(ff0,ff1);
                  tRange = range1f{tt0,tt1};
                  
                  if (dbg)
                    printf(" ... t [%f %f] v [ %f %f ]\n",
                           tRange.lower,
                           tRange.upper,
                           valueRange.lower,
                           valueRange.upper);
                  if (overlaps(self.isoSurface.isoValue)) {
                    intersect(self.isoSurface.isoValue);
                    if (tHit < tMaxL) {
                      return false;
                    }
                  }
                }

                return true;
              },
              /*NO debug:*/false
              );
    if (tHit >= tMaxL) return;

    // tHit is march-local (measured from tEnter). The reported hit distance and
    // depth must be in the original ray parameterization so the iso composites /
    // depth-sorts correctly against other geometry in the shared accel.
    const float tHitOrig = tHit + tEnter;

    // ------------------------------------------------------------------
    // get texture coordinates
    // ------------------------------------------------------------------
    const vec3f osP  = marchOrg + tHit * obj_dir;
    vec3f P  = ti.transformPointFromObjectToWorldSpace(osP);
    // Shading normal from the field gradient. Prefer a sampler-provided analytic
    // gradient (one query, e.g. the AMR basis method); otherwise fall back to a
    // 7-tap central difference.
    vec3f osN;
    float fP;
    if (!isoAnalyticGrad(self.isoSurface.sfSampler, osP, fP, osN)) {
      const float fallbackDelta
        = length(bounds.size()) * .1f
        / float(self.mcGrid.dims.x+self.mcGrid.dims.y+self.mcGrid.dims.z);
      // step matched to the local cell (AMR): low-passes the gradient so fine-
      // scale field detail doesn't turn into noisy, view-dependent dark facets
      const float delta = isoLocalStep(self.isoSurface.sfSampler, osP, fallbackDelta);
      fP         = self.isoSurface.sfSampler.sample(osP);
      float fPx0 = self.isoSurface.sfSampler.sample(osP+vec3f(-delta,0.f,0.f));
      float fPx1 = self.isoSurface.sfSampler.sample(osP+vec3f(+delta,0.f,0.f));
      float fPy0 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,-delta,0.f));
      float fPy1 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,+delta,0.f));
      float fPz0 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,0.f,-delta));
      float fPz1 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,0.f,+delta));
      float dx = 2.f, dy = 2.f, dz = 2.f;
      if (isnan(fPx0)) { dx -= 1.f; fPx0 = fP; }
      if (isnan(fPx1)) { dx -= 1.f; fPx1 = fP; }
      if (isnan(fPy0)) { dy -= 1.f; fPy0 = fP; }
      if (isnan(fPy1)) { dy -= 1.f; fPy1 = fP; }
      if (isnan(fPz0)) { dz -= 1.f; fPz0 = fP; }
      if (isnan(fPz1)) { dz -= 1.f; fPz1 = fP; }
      osN = vec3f(dx == 0.f ? 0.f : (fPx1-fPx0) / dx,
                  dy == 0.f ? 0.f : (fPy1-fPy0) / dy,
                  dz == 0.f ? 0.f : (fPz1-fPz0) / dz);
    }
    // Sensible-default shading normal. A gradient is unusable when it is
    // non-finite (a central-diff tap left the volume / hit a coverage gap, or the
    // hit sample fP was NaN) or (near-)zero (locally flat field). Guarding on the
    // squared length catches all three (NaN/inf/zero fail `l2 > eps && l2 < huge`;
    // note `osN == 0` would miss NaN, which compares unequal to everything and
    // shaded black). Fall back to a ray-facing normal so the pixel is lit.
    const float osN2 = dot(osN,osN);
    if (isnan(fP) || !(osN2 > 1e-20f && osN2 < 1e30f))
      osN = -normalize(obj_dir);
    vec3f n = ti.transformNormalFromObjectToWorldSpace(osN);
    // The field gradient points toward increasing scalar value, i.e. an
    // arbitrary side of the level set (for a field whose high values are
    // enclosed it points *inward*). Used as-is that leaves the visible side of
    // the surface back-facing, so a matte material gets no direct light and a
    // PBR one streaks. An opaque iso-surface is two-sided, so face-forward the
    // shading normal against the ray to always light the side we can see.
    if (dot(n, ray.dir) > 0.f) {
      n   = -n;
      osN = -osN;
    }
    int primID    = ti.getPrimitiveIndex();
    int instID    = ti.getInstanceID();
    
    render::HitAttributes hitData;
    hitData.worldPosition   = ti.transformPointFromObjectToWorldSpace(osP);
    hitData.worldNormal     = normalize(n);
    hitData.objectPosition  = osP;
    hitData.objectNormal    = make_vec4f(normalize(osN));
    hitData.primID          = primID;
    hitData.instID          = instID;
    hitData.t               = tHitOrig;
    hitData.isShadowRay     = ray.isShadowRay;
    float u = 0.f;
    float v = 0.f;
    auto interpolator
      = [u,v,dbg](const GeometryAttribute::DD &attrib,
                  bool faceVarying) -> vec4f
      {
        return vec4f(1.f);
      };
    self.isoSurface.setHitAttributes(hitData,interpolator,world,dbg);

    const DeviceMaterial &material
      = world.materials[self.isoSurface.materialID];
      
    PackedBSDF bsdf
      = material.createBSDF(hitData,world.samplers,dbg);
    float opacity
      = bsdf.getOpacity(ray.isShadowRay,ray.isInMedium,
                        ray.dir,hitData.worldNormal,ray.dbg());
    if (opacity < 1.f) {
      if (rng() > opacity) {
        return;
      }
    }
    material.setHit(ray,hitData,world.samplers,dbg);

    // Commit the hit to the traversal so it clips the ray interval to tHit and
    // culls geometry *behind* the iso surface. Without this the accel never
    // learns the iso hit distance (setHit only writes the PRD), so a farther
    // triangle mesh in the same accel can still run and overwrite the PRD -
    // producing view-/build-dependent depth ordering against other surfaces.
    // Every other primitive (spheres etc.) and the volume isProg report too.
    ti.reportIntersection(tHitOrig, 0);

    // Write hit IDs for AOV channels
    const render::OptixGlobals &globals = render::OptixGlobals::get(ti);
    if (globals.hitIDs) {
      const int rayID
        = ti.getLaunchIndex().x
        + ti.getLaunchDims().x
        * ti.getLaunchIndex().y;
      if (tHitOrig < globals.hitIDs[rayID].depth) {
        globals.hitIDs[rayID].primID = primID;
        globals.hitIDs[rayID].instID
          = globals.world.instIDToUserInstID
          ? globals.world.instIDToUserInstID[instID]
          : instID;
        globals.hitIDs[rayID].objID  = self.isoSurface.userID;
        globals.hitIDs[rayID].depth  = tHitOrig;
      }
    }
  }
  
  template<typename SFSampler>
  inline __rtc_device
  void MCVolumeAccel<SFSampler>::isProg(rtc::TraceInterface &ti)
  {
    const void *pd = ti.getProgramData();
           
    const DD &self = *(typename MCVolumeAccel<SFSampler>::DD*)pd;
    const render::World::DD &world = render::OptixGlobals::get(ti).world;
    // ray in world space
    Ray &ray = *(Ray*)ti.getPRD();
#ifdef NDEBUG
    enum { dbg = false };
#else
    const bool dbg = ray.dbg();
#endif
    
    box3f bounds = self.volume.sfCommon.worldBounds;
    range1f tRange = { ti.getRayTmin(), ti.getRayTmax() };
    
    // ray in object space
    vec3f obj_org = ti.getObjectRayOrigin();
    vec3f obj_dir = ti.getObjectRayDirection();

    auto objRay = ray;
    objRay.org = obj_org;
    objRay.dir = obj_dir;

    if (!boxTest(objRay,tRange,bounds))
      return;
    
    // ------------------------------------------------------------------
    // compute ray in macro cell grid space 
    // ------------------------------------------------------------------
    vec3f mcGridOrigin  = self.mcGrid.gridOrigin;
    vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    vec3f dda_org = obj_org;
    vec3f dda_dir = obj_dir;

    dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    dda_dir = dda_dir * rcp(mcGridSpacing);

    Random rng(ray.rngSeed,hash(ti.getRTCInstanceIndex(),
                                ti.getGeometryIndex(),0));
    // Random rng(ray.rngSeed,hash(ti.getRTCInstanceIndex(),
    //                             ti.getGeometryIndex(),
    //                             ti.getPrimitiveIndex()));
    dda::dda3(dda_org,dda_dir,tRange.upper,
              vec3ui(self.mcGrid.dims),
              [&](const vec3i &cellIdx, float t0, float t1) -> bool
              {
                const float majorant = self.mcGrid.majorant(cellIdx);
                
                if (majorant == 0.f) return true;
                
                vec4f   sample = 0.f;
                range1f tRange = {t0,min(t1,ray.tMax)};
                if (!Woodcock::sampleRange(sample,
                                           self.volume,
                                           obj_org,
                                           obj_dir,
                                           tRange,
                                           majorant,
                                           rng,
                                           dbg)) 
                  return true;
                if (dbg) printf("woodcock hit sample %f %f %f:%f\n",
                                sample.x,
                                sample.y,
                                sample.z,
                                sample.w);
                
                vec3f P_obj = obj_org + tRange.upper * obj_dir;
                vec3f P = ti.transformPointFromObjectToWorldSpace(P_obj);
                ray.setVolumeHit(P,
                                 tRange.upper,
                                 getPos(sample));

                // Write hit IDs for AOV channels on first non-transparent voxel
                const render::OptixGlobals &globals = render::OptixGlobals::get(ti);
                if (globals.hitIDs) {
                  const int rayID
                    = ti.getLaunchIndex().x
                    + ti.getLaunchDims().x
                    * ti.getLaunchIndex().y;
                  if (tRange.upper < globals.hitIDs[rayID].depth) {
                    globals.hitIDs[rayID].primID = ti.getPrimitiveIndex();
                    globals.hitIDs[rayID].instID
                      = globals.world.instIDToUserInstID
                      ? globals.world.instIDToUserInstID[ti.getInstanceID()]
                      : ti.getInstanceID();
                    globals.hitIDs[rayID].objID  = self.volume.userID;
                    globals.hitIDs[rayID].depth  = tRange.upper;
                  }
                }

                ti.reportIntersection(tRange.upper, 0);
                return false;
              },
              /*NO debug:*/false
              );
  }
#endif
}
