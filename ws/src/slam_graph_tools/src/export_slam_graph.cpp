// Read a trusted, same-version SLAM Toolbox snapshot. No optimizer, sensors or TF publishers.
#include <slam_toolbox/serialization.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;

void check_pose(const karto::Pose2 &p) {
  if (!std::isfinite(p.GetX()) || !std::isfinite(p.GetY()) || !std::isfinite(p.GetHeading()))
    throw std::runtime_error("nonfinite graph pose");
}

std::ofstream output(const fs::path &p) {
  std::ofstream out(p, std::ios::binary);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out << std::setprecision(17);
  return out;
}

void export_graph(const std::string &prefix, const fs::path &dir,
                  const rclcpp::Node::SharedPtr &node) {
  for (const auto *extension : {".posegraph", ".data"})
    if (!fs::is_regular_file(prefix + extension) || fs::file_size(prefix + extension) == 0)
      throw std::runtime_error("missing/empty native graph pair");
  if (fs::exists(dir) || fs::is_symlink(dir))
    throw std::runtime_error("output directory already exists");
  // Mapper must be destroyed before Dataset removes registered sensors (same order as SlamToolbox).
  karto::Dataset dataset;
  karto::Mapper mapper;
  if (!serialization::read(prefix, mapper, dataset, node))
    throw std::runtime_error("cannot deserialize graph with installed SLAM Toolbox version");
  if (dataset.GetLasers().size() != 1)
    throw std::runtime_error("expected one RoboRacer LiDAR in graph");
  auto *laser = dynamic_cast<karto::LaserRangeFinder *>(dataset.GetLasers()[0]);
  if (!laser) throw std::runtime_error("invalid serialized laser");
  karto::SensorManager::GetInstance()->RegisterSensor(laser);
  auto scans = mapper.GetAllProcessedScans();
  if (scans.size() < 2) throw std::runtime_error("insufficient processed graph scans");
  std::sort(scans.begin(), scans.end(), [](auto *a, auto *b) {return a->GetTime() < b->GetTime();});
  double previous = 0;
  for (const auto *scan : scans) {
    if (!std::isfinite(scan->GetTime()) || scan->GetTime() <= previous)
      throw std::runtime_error("nonpositive/duplicate graph scan times");
    check_pose(scan->GetCorrectedPose());
    check_pose(scan->GetOdometricPose());
    previous = scan->GetTime();
  }
  // These values match slam_mapping.yaml. Render and trajectory come from this exact snapshot.
  std::unique_ptr<karto::OccupancyGrid> grid(
    karto::OccupancyGrid::CreateFromScans(scans, 0.05, 2, 0.1));
  if (!grid || grid->GetWidth() < 1 || grid->GetHeight() < 1)
    throw std::runtime_error("empty native occupancy grid");
  if (!fs::create_directory(dir)) throw std::runtime_error("cannot exclusively create export directory");
  auto trajectory = output(dir / "trajectory.csv");
  auto odometry = output(dir / "odometry.csv");
  trajectory << "# t,x_m,y_m,yaw_rad\n";
  odometry << "# t,x_m,y_m,yaw_rad\n";
  for (auto *scan : scans) {
    const auto p = scan->GetCorrectedPose(), o = scan->GetOdometricPose();
    trajectory << scan->GetTime() << ',' << p.GetX() << ',' << p.GetY() << ',' << p.GetHeading() << '\n';
    odometry << scan->GetTime() << ',' << o.GetX() << ',' << o.GetY() << ',' << o.GetHeading() << '\n';
  }
  const auto origin = grid->GetCoordinateConverter()->GetOffset();
  auto image = output(dir / "map.pgm");
  image << "P5\n" << grid->GetWidth() << ' ' << grid->GetHeight() << "\n255\n";
  for (int y = grid->GetHeight()-1; y >= 0; --y) {
    for (int x = 0; x < grid->GetWidth(); ++x) {
      auto value = grid->GetValue(karto::Vector2<kt_int32s>(x, y));
      unsigned char pixel = value == karto::GridStates_Free ? 254 :
                            value == karto::GridStates_Occupied ? 0 : 205;
      image.write(reinterpret_cast<const char *>(&pixel), 1);
    }
  }
  auto yaml = output(dir / "map.yaml");
  yaml << "image: map.pgm\nmode: trinary\nresolution: 0.05\norigin: ["
       << origin.GetX() << ", " << origin.GetY()
       << ", 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n";
  auto meta = output(dir / "graph_metadata.json");
  const auto mount = laser->GetOffsetPose();
  check_pose(mount);
  meta << "{\"source\":\"serialized Mapper::GetAllProcessedScans/GetCorrectedPose\","
       << "\"scan_count\":" << scans.size() << ",\"resolution_m\":0.05,"
       << "\"min_pass_through\":2,\"occupancy_threshold\":0.1,"
       << "\"laser_offset_xy_yaw\":[" << mount.GetX() << ',' << mount.GetY() << ',' << mount.GetHeading()
       << "],\"laser_angles_min_max_increment\":[" << laser->GetMinimumAngle() << ','
       << laser->GetMaximumAngle() << ',' << laser->GetAngularResolution() << "]}\n";
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: export_slam_graph <trusted graph prefix> <NEW output directory>\n";
    return 2;
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("offline_slam_graph_export");
  int result = 0;
  try {
    export_graph(argv[1], argv[2], node);
  } catch (const karto::Exception &e) {
    std::cerr << e.GetErrorMessage() << '\n'; result = 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; result = 1;
  }
  rclcpp::shutdown();
  return result;
}
