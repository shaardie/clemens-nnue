#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <exception>
#include <mutex>
#include <queue>
namespace channel {
template <typename T> class Channel {

private:
  std::queue<T> queue;
  std::mutex mtx;
  std::condition_variable not_full;
  std::condition_variable not_empty;
  size_t max_size;
  bool closed;

public:
  Channel(size_t size) : max_size(size), closed(false) {};

  void push(const T &item) {
    if (closed) {
      throw std::exception();
    }
    std::unique_lock<std::mutex> lock(mtx);
    not_full.wait(lock, [this] { return queue.size() < max_size; });
    queue.push(item);
    not_empty.notify_one();
  }

  bool pop(T &item) {
    std::unique_lock<std::mutex> lock(mtx);
    not_empty.wait(lock, [this] { return !queue.empty() || closed; });

    if (queue.empty() && closed) {
      return false;
    }

    item = queue.front();
    queue.pop();
    not_full.notify_one();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mtx);
    closed = true;
    not_empty.notify_all();
  }
};
} // namespace channel
