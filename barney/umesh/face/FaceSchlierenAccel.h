// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/volume/Volume.h"
#include "barney/umesh/common/UMeshField.h"

namespace BARNEY_NS {

  struct FaceSchlierenAccel : public VolumeAccel {
    struct DD {
      const vec3f *vertices;
      const vec3i *indices;
      const vec2i *faceCells;
      const vec3f *cellGradients;
    };

    struct PLD {
      rtc::Geom   *geom     = 0;
      rtc::Group  *group    = 0;
      rtc::Buffer *vertsBuf = 0;
      rtc::Buffer *faceBuf  = 0;
      rtc::Buffer *cellsBuf = 0;
      rtc::Buffer *gradBuf  = 0;
    };
    PLD *getPLD(Device *device) { return &perLogical[device->contextRank()]; }
    std::vector<PLD> perLogical;

    FaceSchlierenAccel(Volume *volume,
                       UMeshField *mesh,
                       GeomTypeCreationFct creatorFct);
    ~FaceSchlierenAccel() override;

    void build(bool full_rebuild) override;

    void extractFaces();

    UMeshField *const mesh;
    GeomTypeCreationFct const creatorFct;

    std::vector<vec3f> h_vertices;
    std::vector<vec3i> h_faces;
    std::vector<vec2i> h_faceCells;
    std::vector<vec3f> h_cellGrads;
    bool extracted = false;
  };

}
