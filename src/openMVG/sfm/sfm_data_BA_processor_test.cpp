#include "testing/testing.h"
#include "sfm_data_BA_processor.hpp"
#include <memory>
#include "ceres/rotation.h"

TEST(BUNDLE_ADJUSTMENT_ADMM, TEST) { EXPECT_TRUE(true); }

TEST(BUNDLE_ADJUSTMENT_ADMM, Optimize) {
    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsic = std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(1920, 1080, 1024, 960, 540);
    openMVG::geometry::Pose3 pose = openMVG::geometry::Pose3();
    std::vector<double> pose_vector = {0.0, 0.0, 0.0, pose.center()(0), pose.center()(1), pose.center()(2)};
    ceres::RotationMatrixToAngleAxis(pose.rotation().data(), pose_vector.data());
    std::vector<double> X = {0.0, 0.0,0.1};
    std::vector<double> x = {1216, 1052};
    openMVG::sfm::Optimize_Options options;
    options.intrinsics_opt = openMVG::cameras::Intrinsic_Parameter_Type::NONE;
    options.extrinsics_opt = openMVG::sfm::Extrinsic_Parameter_Type::NONE;
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

void CreateScene() {
    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsic1 = 
        std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(1920, 1080, 1024, 960, 540);

    std::shared_ptr<openMVG::cameras::IntrinsicBase> intrinsic2 = 
        std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(1920, 1080, 1024, 960, 540);

    openMVG::geometry::Pose3 pose1 = openMVG::geometry::Pose3();
    std::vector<double> pose_vector1 = {0.0, 0.0, 0.0, pose.center()(0), pose.center()(1), pose.center()(2)};
    ceres::RotationMatrixToAngleAxis(pose1.rotation().data(), pose_vector1.data());
    openMVG::geometry::Pose3 pose2 = openMVG::geometry::Pose3();
    std::vector<double> pose_vector2 = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    ceres::RotationMatrixToAngleAxis(pose2.rotation().data(), pose_vector2.data());

    std::vector<double> X1 = {0.0, 0.0,0.1};
    std::vector<double> X2 = {0.0, 0.0,0.1};
    std::vector<double> x1 = {1216, 1052};
    std::vector<double> x2 = {960, 478.53635138612253};

    openMVG::sfm::Optimize_Options options;
    options.intrinsics_opt = openMVG::cameras::Intrinsic_Parameter_Type::NONE;
    options.extrinsics_opt = openMVG::sfm::Extrinsic_Parameter_Type::NONE;
    options.structure_opt = openMVG::sfm::Structure_Parameter_Type::ADJUST_ALL;
    std::shared_ptr<openMVG::sfm::Processor> processor = std::make_shared<openMVG::sfm::CPUProcessor>(
        intrinsic1->getType(),
        intrinsic1->getParams(),
        pose_vector1,
        X1,
        x1,
        options);

   std::shared_ptr<openMVG::sfm::Processor> processor2 = std::make_shared<openMVG::sfm::CPUProcessor>(
        intrinsic2->getType(),
        intrinsic2->getParams(),
        pose_vector2,
        X2,
        x2,
        options);


}
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}