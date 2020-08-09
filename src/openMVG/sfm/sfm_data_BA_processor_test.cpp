#include "testing/testing.h"
#include "sfm_data_BA_processor.hpp"
#include <memory>
#include "ceres/rotation.h"

TEST(BUNDLE_ADJUSTMENT_ADMM, TEST) { EXPECT_TRUE(true); }

TEST(BUNDLE_ADJUSTMENT_ADMM, Optimize) {
    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsic = std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(1920, 1080, 3000, 960, 540);
    openMVG::geometry::Pose3 pose = openMVG::geometry::Pose3();
    std::vector<double> pose_vector = {0.0, 0.0, 0.0, pose.center()(0), pose.center()(1), pose.center()(2)};
    ceres::RotationMatrixToAngleAxis(pose.rotation().data(), pose_vector.data());
    std::vector<double> X = {1.0, 2.0, 3.0};
    std::vector<double> x = {0.0, 0.0};
    openMVG::sfm::Optimize_Options options;
    options.structure_opt = openMVG::sfm::Structure_Parameter_Type::ADJUST_ALL;
    std::shared_ptr<openMVG::sfm::Processor> processor = std::make_shared<openMVG::sfm::CPUProcessor>(
        intrinsic->getType(),
        intrinsic->getParams(),
        pose_vector,
        X,
        x,
        options);

    processor->OptimizeParameters();
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}