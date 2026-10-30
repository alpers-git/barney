// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/common/barney-common.h"
#include "native/volume/MultiVariateGrid.h"
#include "native/Context.h"
#include "native/volume/MCGrid.cuh"
#include <chrono>

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_USER_GEOM(/*file*/MultiVariateGridMC,/*name*/MultiVariateGridMC,
                         MCVolumeAccel<MultiVariateGridSampler>::DD,false,false);

    /*! write and read access to one device's mixed grid, for the mix and
        macro-cell kernels; reads use a point-sampled texture for its cache */
    struct MixedGrid {
#if BARNEY_MV_TEXTURE
      cudaSurfaceObject_t surf;
      cudaTextureObject_t texPoint;
#else
      float *values;
#endif
      vec3i dims;
      int   numComponents;

#if RTC_DEVICE_CODE
      inline __rtc_device void store(const vec3i c, const vec3f v) const
      {
#if BARNEY_MV_TEXTURE
        if (numComponents == 1)
          surf3Dwrite(v.x,surf,c.x*(int)sizeof(float),c.y,c.z);
        else
          surf3Dwrite(::make_float4(v.x,v.y,v.z,0.f),surf,
                      c.x*(int)sizeof(::float4),c.y,c.z);
#else
        float *p = values + index(c)*numComponents;
        p[0] = v.x;
        if (numComponents == 3) { p[1] = v.y; p[2] = v.z; }
#endif
      }

      inline __rtc_device vec3f load(const vec3i c) const
      {
#if BARNEY_MV_TEXTURE
        if (numComponents == 1)
          return vec3f(::tex3D<float>(texPoint,c.x+.5f,c.y+.5f,c.z+.5f),0.f,0.f);
        const ::float4 t = ::tex3D<::float4>(texPoint,c.x+.5f,c.y+.5f,c.z+.5f);
        return vec3f(t.x,t.y,t.z);
#else
        const float *p = values + index(c)*numComponents;
        return numComponents == 1 ? vec3f(p[0],0.f,0.f) : vec3f(p[0],p[1],p[2]);
#endif
      }

      inline __rtc_device size_t index(const vec3i c) const
      { return c.x + (size_t)dims.x*(c.y + (size_t)dims.y*c.z); }
#endif
    };

    __rtc_global void mvGridMix(rtc::ComputeInterface ci,
                                MixedGrid grid,
                                const float *channels,
                                const float *weights,
                                int numChannels)
    {
#if RTC_DEVICE_CODE
      const int v = ci.launchIndex().x;
      const size_t numVoxels = (size_t)grid.dims.x*grid.dims.y*grid.dims.z;
      if ((size_t)v >= numVoxels) return;
      const int nc = grid.numComponents;
      vec3f sum(0.f);
      for (int c=0;c<numChannels;c++) {
        const float w = weights ? weights[c] : 1.f;
        if (w == 0.f) continue;
        const float *x = channels + ((size_t)c*numVoxels + v)*nc;
        sum = sum + w * (nc == 1 ? vec3f(x[0],0.f,0.f) : vec3f(x[0],x[1],x[2]));
      }
      const vec3i c(v % grid.dims.x,
                    (v / grid.dims.x) % grid.dims.y,
                    v / (grid.dims.x*grid.dims.y));
      grid.store(c,sum);
#endif
    }

    /*! macro-cell ranges in two passes, so neighbouring threads read
        neighbouring voxels: first per voxel column over y and z ... */
    __rtc_global void mvGridColumnRanges(rtc::ComputeInterface ci,
                                         MixedGrid grid,
                                         MultiVariateGridField::DD field,
                                         vec3i mcDims,
                                         vec3i cellsPerMC,
                                         range1f *columns)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= grid.dims.x*mcDims.y*mcDims.z) return;
      const int x   = tid % grid.dims.x;
      const int mcy = (tid / grid.dims.x) % mcDims.y;
      const int mcz = tid / (grid.dims.x*mcDims.y);
      const int y0 = max(0,mcy*cellsPerMC.y-1);
      const int y1 = min(grid.dims.y-1,(mcy+1)*cellsPerMC.y+1);
      const int z0 = max(0,mcz*cellsPerMC.z-1);
      const int z1 = min(grid.dims.z-1,(mcz+1)*cellsPerMC.z+1);
      range1f r;
      r.lower = +INFINITY;
      r.upper = -INFINITY;
      for (int z=z0;z<=z1;z++)
        for (int y=y0;y<=y1;y++) {
          const float s = field.reduce(grid.load(vec3i(x,y,z)));
          r.lower = fminf(r.lower,s);
          r.upper = fmaxf(r.upper,s);
        }
      columns[tid] = r;
#endif
    }

    /*! ... then per macro cell over x, overlapping its neighbours by one voxel */
    __rtc_global void mvGridComputeMCs(rtc::ComputeInterface ci,
                                       const range1f *columns,
                                       int dimsX,
                                       int reduction,
                                       MCGrid::DD mcGrid,
                                       vec3i cellsPerMC)
    {
#if RTC_DEVICE_CODE
      const int mcIdx = ci.launchIndex().x;
      if (mcIdx >= mcGrid.dims.x*mcGrid.dims.y*mcGrid.dims.z) return;
      const vec3i mc = mcGrid.cellID(mcIdx);
      const int x0 = max(0,mc.x*cellsPerMC.x-1);
      const int x1 = min(dimsX-1,(mc.x+1)*cellsPerMC.x+1);
      const range1f *col
        = columns + (mc.y + (size_t)mcGrid.dims.y*mc.z)*dimsX;
      float lower = +INFINITY, upper = -INFINITY;
      for (int x=x0;x<=x1;x++) {
        lower = fminf(lower,col[x].lower);
        upper = fmaxf(upper,col[x].upper);
      }
      range1f r;
      if (lower <= upper) {
        r.lower = (reduction == MultiVariateGridField::MAGNITUDE) ? 0.f : lower;
        r.upper = upper;
      }
      mcGrid.scalarRanges[mcIdx] = r;
#endif
    }

    static vec3i cellsPerMC() { return vec3i(4); }

    static MixedGrid mixedGrid(MultiVariateGridField *field, Device *device)
    {
      auto *pld = field->getPLD(device);
      MixedGrid grid;
#if BARNEY_MV_TEXTURE
      grid.surf     = pld->surf;
      grid.texPoint = pld->texPoint;
#else
      grid.values = pld->values;
#endif
      grid.dims = field->dims;
      grid.numComponents = field->numComponents;
      return grid;
    }

    MultiVariateGridField::MultiVariateGridField(Context *context,
                                                 const DevGroup::SP &devices)
      : ScalarField(context,devices)
    { perLogical.resize(devices->numLogical); }

    MultiVariateGridField::~MultiVariateGridField()
    {
      for (auto device : *devices)
        freeMixed(device);
    }

    MultiVariateGridField::PLD *MultiVariateGridField::getPLD(Device *device)
    {
      assert(device && device->contextRank() >= 0);
      assert(device->contextRank() < (int)perLogical.size());
      return &perLogical[device->contextRank()];
    }

    void MultiVariateGridField::allocMixed(Device *device)
    {
      PLD *pld = getPLD(device);
      if (pld->allocDims == dims && pld->allocComponents == numComponents)
        return;
      freeMixed(device);
      SetActiveGPU forDuration(device);
#if BARNEY_MV_TEXTURE
      int maxDim[3];
      BARNEY_CUDA_CALL(DeviceGetAttribute(&maxDim[0],cudaDevAttrMaxSurface3DWidth,
                                          device->rtc->physicalID));
      BARNEY_CUDA_CALL(DeviceGetAttribute(&maxDim[1],cudaDevAttrMaxSurface3DHeight,
                                          device->rtc->physicalID));
      BARNEY_CUDA_CALL(DeviceGetAttribute(&maxDim[2],cudaDevAttrMaxSurface3DDepth,
                                          device->rtc->physicalID));
      if (dims.x > maxDim[0] || dims.y > maxDim[1] || dims.z > maxDim[2])
        throw std::runtime_error
          ("#bn.mv: grid of "+std::to_string(dims.x)+"x"+std::to_string(dims.y)
           +"x"+std::to_string(dims.z)+" exceeds this GPU's 3D surface"
           " limits; rebuild with -DBARNEY_MULTIVARIATE_HW_INTERP=OFF");

      const cudaChannelFormatDesc desc = (numComponents == 1)
        ? cudaCreateChannelDesc<float>()
        : cudaCreateChannelDesc<float4>();
      BARNEY_CUDA_CALL(Malloc3DArray(&pld->array,&desc,
                                     make_cudaExtent(dims.x,dims.y,dims.z),
                                     cudaArraySurfaceLoadStore));
      cudaResourceDesc res;
      memset(&res,0,sizeof(res));
      res.resType         = cudaResourceTypeArray;
      res.res.array.array = pld->array;
      BARNEY_CUDA_CALL(CreateSurfaceObject(&pld->surf,&res));

      cudaTextureDesc tex;
      memset(&tex,0,sizeof(tex));
      tex.addressMode[0]   = cudaAddressModeClamp;
      tex.addressMode[1]   = cudaAddressModeClamp;
      tex.addressMode[2]   = cudaAddressModeClamp;
      tex.filterMode       = cudaFilterModeLinear;
      tex.readMode         = cudaReadModeElementType;
      tex.normalizedCoords = 0;
      BARNEY_CUDA_CALL(CreateTextureObject(&pld->tex,&res,&tex,nullptr));
      tex.filterMode = cudaFilterModePoint;
      BARNEY_CUDA_CALL(CreateTextureObject(&pld->texPoint,&res,&tex,nullptr));
#else
      pld->values = (float*)device->rtc->allocMem
        ((size_t)dims.x*dims.y*dims.z*numComponents*sizeof(float));
#endif
      pld->allocDims       = dims;
      pld->allocComponents = numComponents;
    }

    void MultiVariateGridField::freeMixed(Device *device)
    {
      PLD *pld = getPLD(device);
      SetActiveGPU forDuration(device);
      device->rtc->sync();
#if BARNEY_MV_TEXTURE
      if (pld->tex)   BARNEY_CUDA_CALL_NOTHROW(DestroyTextureObject(pld->tex));
      if (pld->texPoint)
        BARNEY_CUDA_CALL_NOTHROW(DestroyTextureObject(pld->texPoint));
      if (pld->surf)  BARNEY_CUDA_CALL_NOTHROW(DestroySurfaceObject(pld->surf));
      if (pld->array) BARNEY_CUDA_CALL_NOTHROW(FreeArray(pld->array));
#else
      if (pld->values) device->rtc->freeMem(pld->values);
#endif
      *pld = PLD();
    }

    bool MultiVariateGridField::valid() const
    {
      const size_t numVoxels = (size_t)dims.x*dims.y*dims.z;
      const size_t numFloats = numVoxels*numChannels*numComponents;
      return dims.x > 0 && dims.y > 0 && dims.z > 0 && numChannels > 0
        && (numComponents == 1 || numComponents == 3)
        && reduction >= MAGNITUDE && reduction <= DOT
        && component >= 0 && component < numComponents
        && channels && channels->size() >= numFloats*sizeof(float)
        && (!channelWeights
            || channelWeights->size() >= numChannels*sizeof(float));
    }

    void MultiVariateGridField::refresh()
    {
      if (!valid()) {
        std::cerr << "#bn.mv: inconsistent multi-variate grid parameters"
                  << " (dims, numChannels, numComponents, reduction, component"
                  << " or array sizes); not updating" << std::endl;
        return;
      }
      const auto t0 = std::chrono::steady_clock::now();
      const int numVoxels = dims.x*dims.y*dims.z;
      for (auto device : *devices) {
        allocMixed(device);
        auto rtc = device->rtc;
        const float *weights = channelWeights
          ? (const float*)channelWeights->getDD(device) : nullptr;
        __rtc_launch(rtc,mvGridMix,divRoundUp(numVoxels,256),256,
                     mixedGrid(this,device),
                     (const float*)channels->getDD(device),weights,numChannels);
      }
      for (auto device : *devices)
        device->rtc->sync();
      dirty = false;

      if (mcGrid && mcGrid->built()) {
        const vec3i mcDims = mcGrid->dims;
        const int numMC   = mcDims.x*mcDims.y*mcDims.z;
        const int numCols = dims.x*mcDims.y*mcDims.z;
        for (auto device : *devices) {
          auto rtc = device->rtc;
          range1f *columns = (range1f*)rtc->allocMem(numCols*sizeof(range1f));
          __rtc_launch(rtc,mvGridColumnRanges,divRoundUp(numCols,128),128,
                       mixedGrid(this,device),getDD(device),
                       mcDims,cellsPerMC(),columns);
          __rtc_launch(rtc,mvGridComputeMCs,divRoundUp(numMC,128),128,
                       columns,dims.x,reduction,
                       mcGrid->getDD(device),cellsPerMC());
          rtc->sync();
          rtc->freeMem(columns);
        }
        ++mcGrid->contentEpoch;
        for (auto &weak : accels)
          if (auto accel = weak.lock())
            accel->rebuildMajorantsOnly();
      }
      const double ms = std::chrono::duration<double,std::milli>
        (std::chrono::steady_clock::now()-t0).count();
      std::cout << "#bn.mv: mixed " << numChannels << " channels in "
                << ms << " ms" << std::endl;
    }

    MCGrid::SP MultiVariateGridField::buildMCs()
    {
      if (!valid())
        throw std::runtime_error("#bn.mv: cannot build a multi-variate grid"
                                 " from inconsistent parameters");
      MCGrid::SP grid = std::make_shared<MCGrid>(devices);
      const vec3i cpm = cellsPerMC();
      grid->resize(divRoundUp(dims-vec3i(1),cpm));
      grid->gridOrigin  = gridOrigin;
      grid->gridSpacing = vec3f(cpm)*gridSpacing;
      mcGrid = grid;
      refresh();
      return grid;
    }

    MultiVariateGridField::DD MultiVariateGridField::getDD(Device *device)
    {
      DD dd;
      (ScalarField::DD &)dd = ScalarField::getDD(device);
#if BARNEY_MV_TEXTURE
      dd.mixedTex      = (const rtc::TextureObject &)getPLD(device)->tex;
#else
      dd.mixed         = getPLD(device)->values;
#endif
      dd.dims          = dims;
      dd.gridOrigin    = gridOrigin;
      dd.gridSpacing   = gridSpacing;
      dd.numComponents = numComponents;
      dd.reduction     = reduction;
      dd.component     = component;
      dd.direction     = direction;
      return dd;
    }

    MultiVariateGridSampler::DD MultiVariateGridSampler::getDD(Device *device)
    {
      DD dd;
      (MultiVariateGridField::DD &)dd = sf->getDD(device);
      return dd;
    }

    VolumeAccel::SP MultiVariateGridField::createAccel(Volume *volume)
    {
      auto sampler = std::make_shared<MultiVariateGridSampler>(this);
      auto accel = std::make_shared<MCVolumeAccel<MultiVariateGridSampler>>
        (volume,createGeomType_MultiVariateGridMC,sampler);
      accels.push_back(accel);
      return accel;
    }

    IsoSurfaceAccel::SP MultiVariateGridField::createIsoAccel(IsoSurface *)
    { return {}; }

    bool MultiVariateGridField::set1i(const std::string &member, const int &value)
    {
      int *target = nullptr;
      if      (member == "numChannels")   target = &numChannels;
      else if (member == "numComponents") target = &numComponents;
      else if (member == "reduction")     target = &reduction;
      else if (member == "component")     target = &component;
      else return ScalarField::set1i(member,value);
      if (*target != value) { *target = value; dirty = true; }
      return true;
    }

    bool MultiVariateGridField::set3i(const std::string &member, const vec3i &value)
    {
      if (member != "dims") return ScalarField::set3i(member,value);
      if (dims != value) { dims = value; dirty = true; }
      return true;
    }

    bool MultiVariateGridField::set3f(const std::string &member, const vec3f &value)
    {
      vec3f *target = nullptr;
      if      (member == "gridOrigin")  target = &gridOrigin;
      else if (member == "gridSpacing") target = &gridSpacing;
      else if (member == "direction")   target = &direction;
      else return ScalarField::set3f(member,value);
      if (*target != value) { *target = value; dirty = true; }
      return true;
    }

    bool MultiVariateGridField::setString(const std::string &member,
                                          const std::string &value)
    {
      if (member != "reduction") return ScalarField::setString(member,value);
      static const char *names[] = { "magnitude", "component", "dot" };
      for (int r=MAGNITUDE;r<=DOT;r++)
        if (value == names[r]) return set1i(member,r);
      std::cerr << "#bn.mv: unknown reduction '" << value
                << "'; use magnitude, component or dot" << std::endl;
      return true;
    }

    bool MultiVariateGridField::setArray(const std::string &member,
                                         const PODData::SP &data)
    {
      PODData::SP *target = nullptr;
      if      (member == "channels")       target = &channels;
      else if (member == "channelWeights") target = &channelWeights;
      else return false;
      if (*target != data) { *target = data; dirty = true; }
      return true;
    }

    bool MultiVariateGridField::setData(const std::string &member,
                                        const std::shared_ptr<Data> &value)
    {
      if (ScalarField::setData(member,value)) return true;
      return setArray(member,value ? value->as<PODData>() : PODData::SP());
    }

    bool MultiVariateGridField::setObject(const std::string &member,
                                          const Object::SP &value)
    {
      if (auto data = std::dynamic_pointer_cast<PODData>(value))
        if (setArray(member,data)) return true;
      return ScalarField::setObject(member,value);
    }

    void MultiVariateGridField::commit()
    {
      worldBounds.lower = gridOrigin;
      worldBounds.upper = gridOrigin + gridSpacing*vec3f(dims-vec3i(1));
      if (dirty && mcGrid && mcGrid->built())
        refresh();
    }
  }
}
