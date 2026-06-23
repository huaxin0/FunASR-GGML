// Push-to-talk recorder tests.
//
// These tests cover only the input collection state machine. They do not touch
// the microphone or model inference.
#include "pipeline/realtime.hpp"
#include <cstdio>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while(0)

void test_ignores_audio_until_started() {
    printf("\n--- Test 1: ignores audio until started ---\n");

    funasr::PushToTalkRecorder recorder(16000, 30000);
    std::vector<float> samples(160, 0.5f);
    recorder.feed_audio(samples.data(), samples.size());

    TEST_ASSERT(!recorder.is_recording(), "not recording initially");
    TEST_ASSERT(recorder.sample_count() == 0, "no samples before start");
    TEST_ASSERT(recorder.stop().empty(), "stop without start returns empty audio");
}

void test_records_between_start_and_stop() {
    printf("\n--- Test 2: records between start and stop ---\n");

    funasr::PushToTalkRecorder recorder(16000, 30000);
    std::vector<float> samples(320, 0.25f);

    recorder.start();
    recorder.feed_audio(samples.data(), samples.size());
    std::vector<float> audio = recorder.stop();

    TEST_ASSERT(!recorder.is_recording(), "not recording after stop");
    TEST_ASSERT(audio.size() == samples.size(), "returns recorded samples");
    TEST_ASSERT(audio.front() == 0.25f, "preserves sample values");
    TEST_ASSERT(recorder.sample_count() == 0, "buffer is consumed by stop");
}

void test_caps_at_max_duration() {
    printf("\n--- Test 3: caps at max duration ---\n");

    funasr::PushToTalkRecorder recorder(10, 1000);
    std::vector<float> first(8, 0.1f);
    std::vector<float> second(8, 0.2f);

    recorder.start();
    recorder.feed_audio(first.data(), first.size());
    recorder.feed_audio(second.data(), second.size());
    std::vector<float> audio = recorder.stop();

    TEST_ASSERT(audio.size() == 10, "audio is capped at max samples");
    TEST_ASSERT(recorder.was_truncated(), "recorder reports truncation");
    TEST_ASSERT(audio[0] == 0.1f, "keeps earliest samples");
    TEST_ASSERT(audio[9] == 0.2f, "fills until cap");
}

int main() {
    test_ignores_audio_until_started();
    test_records_between_start_and_stop();
    test_caps_at_max_duration();

    printf("\n========================================\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("========================================\n");

    return tests_failed == 0 ? 0 : 1;
}
