// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include "helium/array/Array1D.h"
#include "helium/array/Array3D.h"
#include "helium/array/ObjectArray.h"
// ours
#include "Object.h"
// std
#include <limits>
#include <vector>

namespace BARNEY_NS {
  namespace anari {
    
    struct SpatialField : public Object
    {
      SpatialField(BarneyGlobalState *s);
      ~SpatialField() override;

      static SpatialField *createInstance(std::string_view subtype,
                                          BarneyGlobalState *s);

      void markFinalized() override;

      virtual BNScalarField createBarneyScalarField() const = 0;

      void cleanup()
      {
        if (m_bnField) {
          bnRelease(m_bnField);
          m_bnField = nullptr;
        }
      }

      BNScalarField getBarneyScalarField()
      {
        if (!isValid())
          return {};
        if (!m_bnField)
          m_bnField = createBarneyScalarField();
        return m_bnField;
      }

      virtual box3 bounds() const = 0;

      BNScalarField m_bnField = 0;
    };

    // Subtypes ///////////////////////////////////////////////////////////////////

    struct UnstructuredField : public SpatialField
    {
      UnstructuredField(BarneyGlobalState *s);
      ~UnstructuredField() override;

      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

    private:
      struct Parameters
      {
        Parameters(helium::BaseObject *observer)
          : vertexPosition(observer),
            vertexData(observer),
            cellData(observer),
            index(observer),
            cellType(observer),
            cellBegin(observer)
        {}
        helium::ChangeObserverPtr<helium::Array1D> vertexPosition;
        helium::ChangeObserverPtr<helium::Array1D> vertexData;
        helium::ChangeObserverPtr<helium::Array1D> cellData;
        helium::ChangeObserverPtr<helium::Array1D> index;
        helium::ChangeObserverPtr<helium::Array1D> cellType;
        helium::ChangeObserverPtr<helium::Array1D> cellBegin;
      } m_params;

      struct BarneyData
      {
        BNData vertices{nullptr};
        BNData scalars{nullptr};
        BNData indices{nullptr};
        BNData cellType{nullptr};
        BNData elementOffsets{nullptr};
      } m_bnData;

      box3 m_bounds;
    };

    struct BlockStructuredField : public SpatialField
    {
      BlockStructuredField(BarneyGlobalState *s);
      ~BlockStructuredField() override;
      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;

      struct Parameters
      {
        Parameters(helium::BaseObject *observer)
          : blockDims(observer),
            blockOrigins(observer),
            blockLevel(observer),
            data(observer)
        {}
        helium::ChangeObserverPtr<helium::Array1D> blockDims;
        helium::ChangeObserverPtr<helium::Array1D> blockOrigins;
        helium::ChangeObserverPtr<helium::Array1D> blockLevel;
        helium::ChangeObserverPtr<helium::Array1D> data;
      } m_params;

      struct BarneyData
      {
        BNData scalars{nullptr};
        BNData blockOrigins{nullptr};
        BNData blockDims{nullptr};
        BNData blockLevels{nullptr};
        BNData blockOffsets{nullptr};
      } m_bnData;

      std::vector<uint64_t> m_generatedBlockOffsets;

      box3 m_bounds;
    };

    struct StructuredRegularField : public SpatialField
    {
      StructuredRegularField(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

      math::uint3 m_dims{0u};
      math::float3 m_origin;
      math::float3 m_spacing;
      math::float3 m_coordUpperBound;

      helium::ChangeObserverPtr<helium::Array3D> m_data;
    };

    struct NanoVDBSpatialField : public SpatialField
    {
      NanoVDBSpatialField(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

      std::string m_filter;
      helium::ChangeObserverPtr<helium::Array1D> m_data;

      box3 m_bounds;
      math::float3 m_voxelSize;
    };

    /*! a field of signed, polynomial-modulated radial basis functions -
        each primitive is coeff * x^px y^py z^pz * exp(-exponent*r^2), and the
        field is the signed sum over all primitives covering a point. this is
        the natural representation for gaussian-type atomic orbitals and,
        through the gaussian product theorem, for orbital *pair* products too.

        'groups' are the units a user selects: a group is one orbital, or one
        orbital-pair interaction, and many primitives share one. selection is
        therefore a per-group mask rather than anything per-primitive. */
    struct ParticleRBFField : public SpatialField
    {
      ParticleRBFField(BarneyGlobalState *s);
      ~ParticleRBFField() override;

      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

      struct Parameters
      {
        Parameters(helium::BaseObject *observer)
          : position(observer),
            coefficient(observer),
            exponent(observer),
            cutoff(observer),
            monomial(observer),
            group(observer),
            groupValue(observer),
            groupEnabled(observer),
            polyOffset(observer),
            polyCoeff(observer),
            polyMonomial(observer),
            groupDirection(observer)
        {}
        helium::ChangeObserverPtr<helium::Array1D> position;
        helium::ChangeObserverPtr<helium::Array1D> coefficient;
        helium::ChangeObserverPtr<helium::Array1D> exponent;
        helium::ChangeObserverPtr<helium::Array1D> cutoff;
        helium::ChangeObserverPtr<helium::Array1D> monomial;
        helium::ChangeObserverPtr<helium::Array1D> group;
        helium::ChangeObserverPtr<helium::Array1D> groupValue;
        helium::ChangeObserverPtr<helium::Array1D> groupEnabled;
        /*! polynomial-blob mode: one gaussian per primitive carrying a whole
            polynomial per orbital channel, plus the per-group bond direction
            that makes the field a vector field */
        helium::ChangeObserverPtr<helium::Array1D> polyOffset;
        helium::ChangeObserverPtr<helium::Array1D> polyCoeff;
        helium::ChangeObserverPtr<helium::Array1D> polyMonomial;
        helium::ChangeObserverPtr<helium::Array1D> groupDirection;
      } m_params;

      struct BarneyData
      {
        BNData position{nullptr};
        BNData coefficient{nullptr};
        BNData exponent{nullptr};
        BNData cutoff{nullptr};
        BNData monomial{nullptr};
        BNData group{nullptr};
        BNData groupValue{nullptr};
        BNData groupEnabled{nullptr};
        BNData polyOffset{nullptr};
        BNData polyCoeff{nullptr};
        BNData polyMonomial{nullptr};
        BNData groupDirection{nullptr};
      } m_bnData;

      /*! only used when the app does not supply 'particle.cutoff' - derived
          from the exponents and the relative cutoff threshold */
      std::vector<float> m_generatedCutoffs;

      /*! the primitive arrays never change after the first upload in practice,
          but the filter parameters change on every ui interaction; re-uploading
          a million primitives for a slider drag is what this guards against */
      struct UploadState {
        const void *position{nullptr};
        size_t      count{0};
        bool        done{false};
      } m_uploaded;

      int          m_numChannels{1};
      /*! bit c enables orbital channel c; a uniform, because every channel
          shares one blob's geometry and switching must not rebuild the bvh */
      unsigned     m_channelMask{0xffffffffu};
      int          m_vecMode{0};
      int          m_polyMaxDegree{6};
      /*! -1 leaves barney's own default in place */
      int          m_mcSamplesPerAxis{-1};
      float        m_mcRangePad{-1.f};
      float        m_cutoffThreshold{1e-4f};
      int          m_mcGridSize{0};
      math::float2 m_valueRange{0.f,std::numeric_limits<float>::infinity()};
      bool         m_haveExplicitBounds{false};
      box3         m_explicitBounds;
      box3         m_bounds;
    };

    // Generic wrapper for custom Barney scalar field types
    // This allows ANARI to use fields registered via ScalarFieldRegistry
    struct CustomSpatialField : public SpatialField
    {
      CustomSpatialField(BarneyGlobalState *s, const std::string &type);
      void commitParameters() override;
      void finalize() override;
      void markFinalized() override; // Apply parameters after field is created

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

      void applyParametersToField(); // Apply collected parameters to the Barney field

      std::string m_fieldType;
      box3 m_bounds;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::SpatialField *,
                                    ANARI_SPATIAL_FIELD);
