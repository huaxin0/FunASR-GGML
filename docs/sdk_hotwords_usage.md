# FunASR SDK 热词使用说明

## 需要重新给对方的文件

这次新增了 SDK 函数，所以需要重新编译并重新提供：

- `include/funasr_sdk.h`
- `lib/funasr_sdk.lib`
- `bin/funasr_sdk.dll`

其他运行依赖仍按原来的包提供：

- `ggml.dll`
- `ggml-base.dll`
- `ggml-cpu.dll`
- `ggml-cuda.dll`
- `FunAsr_q8.bin`
- CUDA runtime DLL，如 `cudart64_110.dll`、`cublas64_11.dll`、`cublasLt64_11.dll`

如果对方不重新编译 Qt 程序，只替换 DLL 不能调用新热词函数；要使用新函数，需要用新的 `funasr_sdk.h` 和 `funasr_sdk.lib` 重新编译 Qt 程序。

## 热词文件格式

建议在程序目录放一个 `hotwords.txt`：

```txt
开放时间
FunASR
张三
广东话
```

每行一个热词。SDK 也支持英文逗号、分号、tab 分隔。热词是 UTF-8 文本，不需要音频样本。

## Qt/C++ 调用方式

```cpp
#include "funasr_sdk.h"

FunasrConfig cfg;
funasr_get_default_config(&cfg);
cfg.model_path = "FunAsr_q8.bin";
cfg.use_gpu = 1;
cfg.gpu_id = 0;
cfg.ctx_size = 4096;
cfg.max_new_tokens = 220;

FunasrHandle h = funasr_create();
int rc = funasr_init(h, &cfg);
if (rc != 0) {
    const char* err = funasr_last_error(h);
    // 打印 err
}

// 方式 1：从文件加载热词
rc = funasr_load_hotwords_file(h, "hotwords.txt");
if (rc != 0) {
    const char* err = funasr_last_error(h);
    // 热词加载失败，打印 err；也可以继续无热词识别
}

// 方式 2：直接传 UTF-8 字符串
funasr_set_hotwords(h, "开放时间\nFunASR\n张三");

char text[8192] = {};
FunasrResult result = {};
rc = funasr_transcribe_f32(
    h,
    audio,
    sample_count,
    text,
    sizeof(text),
    &result
);

if (rc >= 0) {
    QString qtext = QString::fromUtf8(text);
}

funasr_destroy(h);
```

## 函数说明

```c
int funasr_set_hotwords(FunasrHandle handle, const char* hotwords_utf8);
```

设置热词文本。`hotwords_utf8` 使用 UTF-8 编码，支持每行一个词，也支持英文逗号、分号、tab 分隔。可以在 `funasr_init` 前或后调用。传空字符串会清空热词。

```c
int funasr_load_hotwords_file(FunasrHandle handle, const char* path_utf8);
```

从文本文件加载热词。建议文件名和路径使用英文，避免 Windows 非 UTF-8 路径兼容问题。返回 `0` 表示成功，返回负数表示失败，可用 `funasr_last_error(handle)` 查看原因。

## 运行目录示例

```txt
QGroundControl.exe
funasr_sdk.dll
ggml.dll
ggml-base.dll
ggml-cpu.dll
ggml-cuda.dll
cudart64_110.dll
cublas64_11.dll
cublasLt64_11.dll
FunAsr_q8.bin
hotwords.txt
```
