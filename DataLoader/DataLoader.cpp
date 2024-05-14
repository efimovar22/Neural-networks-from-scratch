#include "DataLoader.h"

namespace NeuralNetwork {

DataLoader::DataLoader(const std::string &image_path,
                       const std::string &label_path, size_t size_of_batch)
    : batch_size(size_of_batch) {
  image_stream = std::ifstream(image_path, std::ios::binary);
  if (!image_stream.is_open()) {
    throw std::runtime_error("Failed to open image file");
  }

  uint32_t magic_number_image, num_images, num_rows, num_cols;
  image_stream.read(reinterpret_cast<char *>(&magic_number_image),
                    MNIST::POINTER_SIZE);
  image_stream.read(reinterpret_cast<char *>(&num_images), MNIST::POINTER_SIZE);
  image_stream.read(reinterpret_cast<char *>(&num_rows), MNIST::POINTER_SIZE);
  image_stream.read(reinterpret_cast<char *>(&num_cols), MNIST::POINTER_SIZE);

  magic_number_image = __builtin_bswap32(magic_number_image);
  num_images = __builtin_bswap32(num_images);
  num_rows = __builtin_bswap32(num_rows);
  num_cols = __builtin_bswap32(num_cols);

  image_size = num_rows * num_cols;
  if (image_size != MNIST::IMAGE_SIZE) {
    throw std::runtime_error("Incorrect file format");
  }

  if (magic_number_image != MNIST::MAGIC_NUMBER_IMAGE) {
    throw std::runtime_error("Incorrect magic number for images");
  }

  label_stream = std::ifstream(label_path, std::ios::binary);
  if (!label_stream.is_open()) {
    throw std::runtime_error("Failed to open label file");
  }

  uint32_t magic_number_label, num_labels;
  label_stream.read(reinterpret_cast<char *>(&magic_number_label),
                    MNIST::POINTER_SIZE);
  label_stream.read(reinterpret_cast<char *>(&num_labels), MNIST::POINTER_SIZE);
  magic_number_label = __builtin_bswap32(magic_number_label);

  if (magic_number_label != MNIST::MAGIC_NUMBER_LABEL) {
    throw std::runtime_error("Incorrect magic number for labels");
  }
  current_index = 0;
}

void DataLoader::LoadNextBatch(Batch &batch) {
  size_t batch_limit = std::min(num_images - current_index, batch_size);
  batch.clear();
  batch.reserve(batch_limit);
  for (size_t i = 0; i < batch_limit; ++i) {
    batch.emplace_back(ExtractImage(), MNIST::ConvertInt(ExtractLabel()));
    current_index++;
  }
}

void DataLoader::Restart() {
  current_index = 0;
  image_stream.seekg(4 * MNIST::POINTER_SIZE, std::ios::beg);
  label_stream.seekg(2 * MNIST::POINTER_SIZE, std::ios::beg);
}

Eigen::Vector<double, MNIST::IMAGE_SIZE> DataLoader::ExtractImage() {
  if (current_index >= num_images) {
    throw std::runtime_error("Index out of range");
  }

  Vector image_data(MNIST::IMAGE_SIZE);
  for (int32_t i = 0; i < image_size; ++i) {
    uint8_t pixel = 0;
    image_stream.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
    image_data(i) = static_cast<double>(pixel) / MNIST::PIXEL_MAX;
  }
  return image_data;
}

uint8_t DataLoader::ExtractLabel() {
  if (current_index >= num_images) {
    throw std::runtime_error("Index out of range");
  }
  uint8_t label = 0;
  label_stream.read(reinterpret_cast<char *>(&label), sizeof(label));
  return label;
}

} // namespace NeuralNetwork
