// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Renderer.h"

namespace barney_device {

Renderer::Renderer(BarneyGlobalState *s)
    : Object(ANARI_RENDERER, s), m_backgroundImage(this)
{
  barneyRenderer = bnRendererCreate(deviceState()->tether->context, "default");
}

Renderer::~Renderer()
{
  bnRelease(barneyRenderer);
}

void Renderer::commitParameters()
{
  m_pixelSamples = getParam<int>("pixelSamples", 1);
  m_ambientRadiance = getParam<float>("ambientRadiance", 1.f);
  m_aoRadius = getParam<float>("aoRadius", getParam<float>("AORadius", 0.f));
  m_aoSamples = getParam<int>("aoSamples", getParam<int>("AOSamples", 0));
  m_crosshairs = getParam<bool>("crosshairs", false);
  m_denoise = getParam<bool>("denoise", true);
  m_fadeOutDenoiser = getParam<bool>("fadeOutDenoiser", true);
  m_upscale = getParam<bool>("upscale", false);
  m_background = getParam<math::float4>("background", math::float4(0, 0, 0, 1));
  m_backgroundImage = getParamObject<Array2D>("background");
  m_cutPlane = getParam<math::float4>("cutPlane", math::float4(0, 0, 0, 0));
  m_gladstoneDale = getParam<float>("gladstoneDale", 1e-4f);
  m_knife = getParam<math::float3>("knife", math::float3(1, 0, 0));
  m_schlierenOpacity = getParam<float>("schlierenOpacity", 0.9f);
  m_schlierenRange = getParam<math::float2>("schlierenRange", math::float2(-1, 1));
}

void Renderer::finalize()
{
  bnSetVec(barneyRenderer, "bgColor", m_background);
  bnSet1i(barneyRenderer, "crosshairs", (int)m_crosshairs);
  bnSet1i(barneyRenderer, "pathsPerPixel", (int)m_pixelSamples);
  bnSet1f(barneyRenderer, "ambientRadiance", m_ambientRadiance);
  bnSet1f(barneyRenderer, "aoRadius", m_aoRadius);
  bnSet1i(barneyRenderer, "aoSamples", m_aoSamples);
  bnSet4f(barneyRenderer, "cutPlane",
          m_cutPlane.x, m_cutPlane.y, m_cutPlane.z, m_cutPlane.w);
  bnSet1f(barneyRenderer, "gladstoneDale", m_gladstoneDale);
  bnSet3f(barneyRenderer, "knife", m_knife.x, m_knife.y, m_knife.z);
  bnSet1f(barneyRenderer, "schlierenOpacity", m_schlierenOpacity);
  bnSet4f(barneyRenderer, "schlierenRange",
          m_schlierenRange.x, m_schlierenRange.y, 0.f, 0.f);

  if (m_backgroundImage) {
    int sx = m_backgroundImage->size().x;
    int sy = m_backgroundImage->size().y;
    const bn_float4 *texels
      = (const bn_float4 *)m_backgroundImage->data();
    barneyBackgroundImage
      = bnTexture2DCreate(deviceState()->tether->context,-1,
                          BN_FLOAT4,sx,sy,
                          texels,
                          BN_TEXTURE_LINEAR,
                          BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP);
    bnSetObject(barneyRenderer,"bgTexture",barneyBackgroundImage);
  } else {
    if (barneyBackgroundImage) {
      bnRelease(barneyBackgroundImage);
      barneyBackgroundImage = 0;
      bnSetObject(barneyRenderer,"bgTexture",0);
    }
  }
  bnCommit(barneyRenderer);
}

bool Renderer::crosshairs() const
{
  return m_crosshairs;
}

bool Renderer::denoise() const
{
  return m_denoise;
}

bool Renderer::fadeOutDenoiser() const
{
  return m_fadeOutDenoiser;
}

bool Renderer::upscale() const
{
  return m_upscale;
}

bool Renderer::isValid() const
{
  return barneyRenderer != 0;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Renderer *);
