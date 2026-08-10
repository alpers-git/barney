// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <array>

#include "barney/volume/MCGrid.h"
#include "barney/volume/Volume.h"

namespace BARNEY_NS {

  struct Volume;
  struct VolumeAccel;
  struct IsoSurface;
  struct IsoSurfaceAccel;
  struct ModelSlot;
  struct ScalarFieldSampler;

  /*! abstracts any sort of scalar field (unstructured, amr,
    structured, rbfs....) _before_ any transfer function(s) get
    applied to it */
  struct ScalarField : public barney_api::ScalarField
  {
    typedef std::shared_ptr<ScalarField> SP;

    /*! Device-side data common to all ScalarFields that live on the device */
    struct DD {
      /*! world bounds, CLIPPED TO DOMAIN (if non-empty domain is present!) */
      box3f                worldBounds;
    };
    DD getDD(Device *device) const { return { worldBounds }; }
    
    ScalarField(Context *context,
                const DevGroup::SP &devices,
                const box3f &domain=box3f());

    static ScalarField::SP create(Context *context,
                                  const DevGroup::SP &devices,
                                  const std::string &type);

    /*! creates an acceleration structure for a 'volume' object using
        this scalar field type */
    virtual std::shared_ptr<VolumeAccel>
    createAccel(Volume *volume) = 0;

    /*! creates an acceleration structure for a 'isoSurface' geometry
        using this scalar field type */
    virtual std::shared_ptr<IsoSurfaceAccel>
    createIsoAccel(IsoSurface *isoSurface)
    { return {}; }

    MCGrid::SP getMCs()
    {
      if (!mcGrid)
        mcGrid = buildMCs();
      else if (!mcGridValid)
        refillMCs(*mcGrid);
      mcGridValid = true;
      return mcGrid;
    }

    /*! create, fill, and return a macrocell grid for this field */
    virtual MCGrid::SP buildMCs();

    /*! re-rasterize the per-macrocell scalar ranges of an already sized
        grid, after this field's scalars changed but its geometry did
        not. The grid keeps its identity - a MajorantsGrid built over it
        holds it by pointer - and only its cell ranges are recomputed.
        Field types that don't override this keep the ranges they had. */
    virtual void refillMCs(MCGrid &) {}

    /*! tell the next getMCs() that the cached macrocell ranges describe
        stale scalars. Called from commit(), since the ranges drive both
        the volume majorants and the iso-surface's per-macrocell
        rejection test: a grid left over from a previously set scalar
        renders the new one wrong (and, because the majorants still
        bound the old data, often looks like nothing changed at all). */
    void invalidateMCs() { mcGridValid = false; }

    MCGrid::SP  mcGrid;
    bool        mcGridValid = false;

    /*! the one sampler every accel over this field shares. A sampler owns a
        cuBQL bvh over the field's elements, which for a large mesh is the
        biggest derived structure there is; giving the volume accel and the
        iso-surface accel one each would build and hold two of them, and
        switching between the two presentations would transiently need both.
        The bvh is built from element bounds only, so it survives a scalar
        swap and one instance serves every accel. Created on first use by the
        concrete field's getSampler(). */
    std::shared_ptr<ScalarFieldSampler> sampler;
    box3f       worldBounds;
    
    /*! a clipping box used to restrict whatever primitives the volume
        may be made up to down to a specific 3d box. e.g., if there's
        ghost cells, or if this is a spatial partitioning of a umesh,
        etc */
    const box3f domain;
    DevGroup::SP const devices;
  };
  
  /*! abstraction for a class that can sample a given scalar
    field. it's up to that class to create the right sampler for its
    data, and to do that only for the kind of traversers/accels that
    actually need to be able to sample.

    For the device side, all the actual sampling functionality will be
    in the DD's of the derived classes; the parent class doesnt' even
    have a DD, because all the device-side sampling code will (have to
    be) resolved through templates, in which case the classes using
    the sampler will know the actual type of that sampler (and it's
    DD)
  */
  struct ScalarFieldSampler {
    virtual void build() = 0;
    struct DD {
      /* derived classes ned to implement:
         
         inline __both__ float sample(vec3f P, bool dbg)
         
      */
    };
  };
  
}

