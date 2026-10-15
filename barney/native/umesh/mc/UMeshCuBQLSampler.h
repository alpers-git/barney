// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/umesh/common/UMeshField.h"
#include "native/common/CuBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {
  namespace native {
    
    /*! a umesh scalar field, with a CuBQL bvh sampler */
    struct UMeshCuBQLSampler : public ScalarFieldSampler {
      using bvh_t  = cuBQL::BinaryBVH<float,3>;
      using node_t = typename bvh_t::Node;
    
      struct DD : public UMeshField::DD {
        inline __rtc_device float sample(vec3f P, bool dbg = false) const;
      
        bvh_t bvh;
      };
      DD getDD(Device *device);
    
      /*! per-device data - parent store the umesh, we just store the
        bvh nodes */
      struct PLD {
        bvh_t bvh = { 0,0,0,0 };
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;
    
      UMeshCuBQLSampler(UMeshField *mesh);
    
      /*! builds the string that allows for properly matching optix
        device progs for this type */
      inline static std::string typeName() { return "UMesh_CuBQL"; }

      void build() override;

      UMeshField *const mesh;
      const DevGroup::SP devices;
    };
  
    inline __rtc_device
    float UMeshCuBQLSampler::DD::sample(vec3f P, bool dbg) const
    {
      typename bvh_t::box_t box; box.lower = box.upper = to_cubql(P);

      /* numChannels>0 is the orbital-current field: the sample is
         |sum_{c enabled} j_c(P)|, one scalar, which the volume integrator then
         puts through the transfer function once. The channels are never
         classified separately and their colours are never blended - that would
         composite N colours instead of applying the colour map to the one
         physically meaningful number. */
      if (numChannels > 0) {
        float retVal = 0.f;
        auto lambda = [this,P,&retVal](const uint32_t primID)
        {
          if (this->tetVecMag(retVal,primID,P))
            return CUBQL_TERMINATE_TRAVERSAL;
          return CUBQL_CONTINUE_TRAVERSAL;
        };
        cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
        return retVal;
      }

      float retVal = NAN;
      auto lambda = [this,P,&retVal,dbg]
        // __rtc_device
        (const uint32_t primID)
      {
        if (this->eltScalar(retVal,primID,P,dbg))
          return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
      cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
      return retVal;
    }

  }
} // ::BARNEY_NS


