// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "barney/umesh/face/FaceSchlierenAccel.h"

#include <unordered_map>
#include <array>
#include <cstdint>
#include <iostream>

namespace BARNEY_NS {

  namespace {
    template<typename T>
    std::vector<T> downloadPOD(const PODData::SP &d, Device *dev)
    {
      std::vector<T> h(d->count);
      d->download(dev, h.data());
      return h;
    }

    // gradient of the linear interpolant over a tet: [e1;e2;e3] g = ds
    inline vec3f tetGradient(vec3f p0, vec3f p1, vec3f p2, vec3f p3,
                             float s0, float s1, float s2, float s3)
    {
      vec3f e1 = p1 - p0, e2 = p2 - p0, e3 = p3 - p0;
      vec3f ds(s1 - s0, s2 - s0, s3 - s0);
      // M has rows e1,e2,e3; solve M g = ds via cofactors
      float det =
          e1.x * (e2.y * e3.z - e2.z * e3.y) -
          e1.y * (e2.x * e3.z - e2.z * e3.x) +
          e1.z * (e2.x * e3.y - e2.y * e3.x);
      if (det == 0.f) return vec3f(0.f);
      float rdet = 1.f / det;
      // inverse rows (M^{-1}) applied to ds
      vec3f c0(e2.y * e3.z - e2.z * e3.y,
               e1.z * e3.y - e1.y * e3.z,
               e1.y * e2.z - e1.z * e2.y);
      vec3f c1(e2.z * e3.x - e2.x * e3.z,
               e1.x * e3.z - e1.z * e3.x,
               e1.z * e2.x - e1.x * e2.z);
      vec3f c2(e2.x * e3.y - e2.y * e3.x,
               e1.y * e3.x - e1.x * e3.y,
               e1.x * e2.y - e1.y * e2.x);
      return vec3f(dot(c0, ds), dot(c1, ds), dot(c2, ds)) * rdet;
    }

    struct FaceKey {
      int a, b, c;
      bool operator==(const FaceKey &o) const { return a == o.a && b == o.b && c == o.c; }
    };
    struct FaceKeyHash {
      std::size_t operator()(const FaceKey &k) const
      {
        std::size_t h = 1469598103934665603ull;
        for (int v : {k.a, k.b, k.c}) { h ^= (std::size_t)v; h *= 1099511628211ull; }
        return h;
      }
    };
    inline FaceKey sortedKey(int a, int b, int c)
    {
      if (a > b) std::swap(a, b);
      if (b > c) std::swap(b, c);
      if (a > b) std::swap(a, b);
      return FaceKey{a, b, c};
    }
  } // anonymous

  FaceSchlierenAccel::FaceSchlierenAccel(Volume *volume,
                                         UMeshField *mesh,
                                         GeomTypeCreationFct creatorFct)
    : VolumeAccel(volume), mesh(mesh), creatorFct(creatorFct)
  {
    perLogical.resize(devices->numLogical);
  }

  FaceSchlierenAccel::~FaceSchlierenAccel()
  {
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      if (pld->group)    { device->rtc->freeGroup(pld->group);   pld->group = 0; }
      if (pld->geom)     { device->rtc->freeGeom(pld->geom);     pld->geom = 0; }
      if (pld->vertsBuf) { device->rtc->freeBuffer(pld->vertsBuf); pld->vertsBuf = 0; }
      if (pld->faceBuf)  { device->rtc->freeBuffer(pld->faceBuf);  pld->faceBuf = 0; }
      if (pld->cellsBuf) { device->rtc->freeBuffer(pld->cellsBuf); pld->cellsBuf = 0; }
      if (pld->gradBuf)  { device->rtc->freeBuffer(pld->gradBuf);  pld->gradBuf = 0; }
    }
  }

  void FaceSchlierenAccel::extractFaces()
  {
    if (extracted) return;
    Device *dev = (*devices)[0];
    SetActiveGPU forDuration(dev);

    h_vertices        = downloadPOD<vec3f>(mesh->vertices, dev);
    auto scalars      = downloadPOD<float>(mesh->scalars, dev);
    auto indices      = downloadPOD<int>(mesh->indices, dev);
    auto cellOffsets  = downloadPOD<int>(mesh->cellOffsets, dev);
    auto cellTypes    = downloadPOD<uint8_t>(mesh->cellTypes, dev);
    dev->sync();

    const int numCells = (int)mesh->cellOffsets->count;
    h_cellGrads.assign(numCells, vec3f(0.f));

    if (!mesh->scalarsArePerVertex) {
      std::cout << OWL_TERMINAL_YELLOW
                << "#bn.faceSchlieren: WARNING scalars are per-cell, but "
                   "face-iteration needs per-vertex scalars to reconstruct the "
                   "gradient; the schlieren field will be empty. Use per-vertex "
                   "data or the marching schlieren mode."
                << OWL_TERMINAL_DEFAULT << std::endl;
    }

    std::unordered_map<FaceKey, int, FaceKeyHash> faceMap;
    faceMap.reserve(numCells * 2);

    int numNonTet = 0;
    for (int c = 0; c < numCells; ++c) {
      const uint8_t t = cellTypes[c];
      if (t != _VTK_TET && t != _ANARI_TET) { ++numNonTet; continue; } // tets only
      const int ofs = cellOffsets[c];
      const int iv[4] = { indices[ofs + 0], indices[ofs + 1],
                          indices[ofs + 2], indices[ofs + 3] };
      const vec3f p[4] = { h_vertices[iv[0]], h_vertices[iv[1]],
                           h_vertices[iv[2]], h_vertices[iv[3]] };
      h_cellGrads[c] = tetGradient(p[0], p[1], p[2], p[3],
                                   scalars[iv[0]], scalars[iv[1]],
                                   scalars[iv[2]], scalars[iv[3]]);

      // each face drops one vertex; the dropped vertex is 'opposite'
      const int fv[4][3] = { {iv[1], iv[2], iv[3]}, {iv[0], iv[2], iv[3]},
                             {iv[0], iv[1], iv[3]}, {iv[0], iv[1], iv[2]} };
      const int opp[4]   = { iv[0], iv[1], iv[2], iv[3] };
      for (int f = 0; f < 4; ++f) {
        const int a = fv[f][0], b = fv[f][1], cc = fv[f][2];
        const vec3f N = cross(h_vertices[b] - h_vertices[a],
                              h_vertices[cc] - h_vertices[a]);
        const bool tetOnPlus = dot(h_vertices[opp[f]] - h_vertices[a], N) > 0.f;
        const FaceKey key = sortedKey(a, b, cc);
        auto it = faceMap.find(key);
        if (it == faceMap.end()) {
          const int fi = (int)h_faces.size();
          h_faces.push_back(vec3i(a, b, cc));
          vec2i cells(-1, -1);
          (tetOnPlus ? cells.y : cells.x) = c;
          h_faceCells.push_back(cells);
          faceMap.emplace(key, fi);
        } else {
          vec2i &cells = h_faceCells[it->second];
          (tetOnPlus ? cells.y : cells.x) = c;
        }
      }
    }
    if (numNonTet > 0)
      std::cout << OWL_TERMINAL_YELLOW
                << "#bn.faceSchlieren: WARNING " << numNonTet << " of " << numCells
                << " cells are not tetrahedra and were skipped; face-iteration "
                   "schlieren only supports tets, so this image is missing "
                   "those elements. Use the marching schlieren mode for "
                   "non-tet meshes."
                << OWL_TERMINAL_DEFAULT << std::endl;
    extracted = true;
  }

  void FaceSchlierenAccel::build(bool /*full_rebuild*/)
  {
    extractFaces();
    if (h_faces.empty()) return;

    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        pld->vertsBuf = device->rtc->createBuffer(h_vertices.size() * sizeof(vec3f),
                                                  h_vertices.data());
        pld->faceBuf  = device->rtc->createBuffer(h_faces.size() * sizeof(vec3i),
                                                  h_faces.data());
        pld->cellsBuf = device->rtc->createBuffer(h_faceCells.size() * sizeof(vec2i),
                                                  h_faceCells.data());
        pld->gradBuf  = device->rtc->createBuffer(h_cellGrads.size() * sizeof(vec3f),
                                                  h_cellGrads.data());
        rtc::GeomType *gt = device->geomTypes.get(creatorFct);
        pld->geom = gt->createGeom();
        pld->geom->setVertices(pld->vertsBuf, (int)h_vertices.size());
        pld->geom->setIndices(pld->faceBuf, (int)h_faces.size());
      }
      DD dd;
      dd.vertices      = (const vec3f *)pld->vertsBuf->getDD();
      dd.indices       = (const vec3i *)pld->faceBuf->getDD();
      dd.faceCells     = (const vec2i *)pld->cellsBuf->getDD();
      dd.cellGradients = (const vec3f *)pld->gradBuf->getDD();
      pld->geom->setDD(&dd);

      if (!pld->group)
        pld->group = device->rtc->createTrianglesGroup({pld->geom});
      pld->group->buildAccel();

      Volume::PLD *volumePLD = volume->getPLD(device);
      if (volumePLD->generatedGroups.empty())
        volumePLD->generatedGroups = { pld->group };
    }
  }

}
