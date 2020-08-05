// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2020 junlinp Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_DATA_BA_ADMM_HPP
#define OPENMVG_SFM_SFM_DATA_BA_ADMMHPP
#include <memory>

#include "openMVG/sfm/sfm_data_BA.hpp"
namespace ceres {
class CostFunction;
}
namespace openMVG {
namespace cameras {
struct IntrinsicBase;
}
}  // namespace openMVG
namespace openMVG {
namespace sfm {
struct SfM_Data;
}
}  // namespace openMVG

namespace openMVG {
namespace sfm {
class Bundle_Adjustment_Admm : public Bundle_Adjustment {
 public:
  struct BA_Admm_options {};

 private:
  BA_Admm_options admm_options_;

 public:
  explicit Bundle_Adjustment_Admm(
      const Bundle_Adjustment_Admm::BA_Admm_options options =
          std::move(Bundle_Adjustment_Admm::BA_Admm_options()));
  bool Adjust(sfm::SfM_Data& sfm_data,
              const Optimize_Options& options) override;
};  // class Bundle_adjustment_Admm
}  // namespace sfm
}  // namespace openMVG
#endif  //  OPENMVG_SFM_SFM_DATA_BA_ADMMHPP