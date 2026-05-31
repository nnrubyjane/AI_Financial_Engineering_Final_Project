# AI Financial Engineering Final Project

Final project for the university course **Artificial Intelligence and Financial Engineering**.

This project is educational and beginner-friendly. It is not a production trading system. This project is an educational small-scale GPT-2 style implementation. It includes the core GPT-2 components such as token embedding, positional embedding, masked multi-head self-attention, feed-forward network, transformer block, and text generation. The purpose is to demonstrate the structure clearly enough for function-by-function explanation in an oral test.

## Project Overview

This repository has two parts:

1. **Mock automatic trading system**
   - Based on the concept of Korea Investment Securities API, also called **한국투자증권 API** or **KIS API**.
   - Runs in mock mode by default.
   - Uses fake stock prices.
   - Prints simulated `BUY`, `SELL`, or `HOLD` signals.
   - Never places a real order by default.

2. **Mini GPT-2 demo**
   - Uses Python and PyTorch.
   - Builds a very small GPT-style transformer.
   - Trains quickly on a tiny text string.
   - Generates a short sample of text.

## File Structure

```text
AI_Financial_Engineering_Final_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── trading_system/
│   ├── __init__.py
│   ├── config.py
│   ├── kis_auth.py
│   ├── kis_market.py
│   ├── strategy.py
│   └── main.py
│
├── gpt2_demo/
│   ├── __init__.py
│   ├── model.py
│   ├── train_demo.py
│   └── generate_demo.py
│
└── results/
    ├── trading_sample_output.txt
    └── gpt_sample_output.txt
```

## Installation

Use this command from the project root:

```bash
python -m pip install -r requirements.txt
```

If your computer needs the CPU-only PyTorch package, use the installation command recommended on the official PyTorch website for your operating system.

Do **not** create or commit a real `.env` file for this project. The included `.env.example` contains fake placeholder values only.

## How To Run

Run the mock trading system:

```bash
python -m trading_system.main
```

Run the GPT training demo:

```bash
python -m gpt2_demo.train_demo
```

Run the GPT text generation demo:

```bash
python -m gpt2_demo.generate_demo
```

## Trading System Explanation

### What Is An API?

An **API** means Application Programming Interface. It is a way for one program to request data or services from another program.

In finance, an API can let a program request stock prices, check account information, or place orders. This project only demonstrates the idea and uses mock data.

### What Is KIS API?

**KIS API** means Korea Investment Securities API, or **한국투자증권 API**. A real KIS API can be used to connect software with Korea Investment Securities services.

In this project, KIS API is only used as an educational concept:

- authentication concept
- access token concept
- market price request concept
- trading signal concept

No real KIS API request is made in the default program.

### What Is An Access Token?

An access token is like a temporary permission pass. After a real API checks the app key and app secret, it may return an access token. The program then sends that token when requesting data.

In mock mode, this project returns:

```text
mock_access_token
```

This is fake and safe.

### What Is Mock Mode?

Mock mode means the project uses fake data instead of connecting to a real API.

In `trading_system/config.py`, the default is:

```python
MOCK_MODE = True
```

Because mock mode is enabled, the project can run without real API keys.

### What Is Dry-Run Mode?

Dry-run mode means the program only shows what it would do. It does not place real orders.

In `trading_system/config.py`, the default is:

```python
DRY_RUN = True
```

### Trading Strategy

The strategy uses a moving average:

1. Calculate the average of the most recent prices.
2. Compare the current price with the moving average.
3. Print a signal:
   - `BUY`: current price is lower than the moving average by more than the threshold.
   - `SELL`: current price is higher than the moving average by more than the threshold.
   - `HOLD`: current price is close to the moving average.

The threshold is `0.02`, which means 2%.

## Trading Files: Function By Function

### `trading_system/config.py`

- `read_bool_env(name, default)`
  - Reads optional environment variables such as `MOCK_MODE`.
  - Converts text like `true` or `false` into a Python boolean.

- `validate_safety_settings()`
  - Checks that dangerous settings are not accidentally enabled.
  - Helps keep this project safe for public upload and classroom use.

Important values:

- `MOCK_MODE = True`
- `DRY_RUN = True`
- `REAL_TRADING_ENABLED = False`
- `STOCK_SYMBOL = "005930"`

### `trading_system/kis_auth.py`

- `get_access_token()`
  - If mock mode is enabled, returns `mock_access_token`.
  - If real mode is enabled, calls the safe template function.

- `request_real_access_token_template()`
  - Explains how real token authentication would work.
  - Does not make a real request.
  - Does not include real secrets.

### `trading_system/kis_market.py`

- `get_stock_price(access_token, symbol)`
  - Gets price data.
  - Uses fake sample prices in mock mode.

- `get_mock_stock_price(symbol)`
  - Returns sample price history and current price.

- `request_real_stock_price_template(access_token, symbol)`
  - Shows the structure of a real market price request.
  - Stays disabled for safety.

### `trading_system/strategy.py`

- `calculate_moving_average(prices, window)`
  - Calculates the average of the latest prices.

- `generate_signal(current_price, moving_average, threshold_percent)`
  - Returns `BUY`, `SELL`, or `HOLD`.
  - Also returns a simple explanation.

### `trading_system/main.py`

- `build_output_text(...)`
  - Formats the result for the console.

- `save_sample_output(output_text)`
  - Saves the output into `results/trading_sample_output.txt`.

- `run_trading_demo()`
  - Runs the full workflow:
    - safety check
    - mock access token
    - mock price request
    - moving average
    - signal generation
    - output saving

## GPT-2 Explanation

### What Is GPT?

GPT means Generative Pre-trained Transformer. It predicts the next token from previous tokens. Repeating that prediction many times creates generated text.

This project uses characters as tokens to keep the demo small.

### What Is A Transformer?

A transformer is a neural network architecture that uses attention. Attention lets the model decide which earlier tokens are important for predicting the next token.

### Token Embedding

Token embedding converts token IDs into vectors. A vector is a list of numbers that the neural network can learn from.

### Positional Embedding

A transformer needs position information because attention alone does not know token order. Positional embedding adds information such as "this is character position 1" or "this is character position 2".

### Q, K, V

In attention:

- `Q` means Query: what the current token is looking for.
- `K` means Key: what each token offers.
- `V` means Value: the information copied after attention chooses where to look.

### Masked Self-Attention

Masked self-attention prevents the model from seeing future tokens. This is important because GPT predicts the next token using only previous tokens.

Example: when predicting character 5, the model should not look at character 6.

### Multi-Head Attention

Multi-head attention means the model uses several attention heads at the same time. Each head can learn a different pattern.

### Feed-Forward Network

The feed-forward network is a small neural network inside each transformer block. It processes the information after attention.

### Transformer Block

A transformer block contains:

1. Layer normalization
2. Masked multi-head self-attention
3. Residual connection
4. Layer normalization
5. Feed-forward network
6. Residual connection

### Text Generation

The model receives a prompt, predicts the next character, adds that character to the prompt, then repeats the process.

## GPT Files: Function By Function

### `gpt2_demo/model.py`

- `CharTokenizer`
  - Converts text to character IDs.
  - Converts character IDs back to text.

- `MultiHeadSelfAttention`
  - Builds Q, K, and V.
  - Applies a future-token mask.
  - Combines multiple attention heads.

- `FeedForward`
  - Adds a small neural network after attention.

- `TransformerBlock`
  - Combines attention, feed-forward network, layer normalization, and residual connections.

- `MiniGPT`
  - Combines token embedding, positional embedding, transformer blocks, final layer normalization, and output layer.
  - Includes `generate()` for text generation.

### `gpt2_demo/train_demo.py`

- `create_model_and_tokenizer()`
  - Creates the tokenizer and small model.

- `get_batch(data, block_size, batch_size)`
  - Creates small random training examples from the tiny text.

- `train_model(steps=80, show_loss=True)`
  - Trains for a small number of steps.
  - Prints loss so we can see training progress.

- `main()`
  - Runs the training demo.

### `gpt2_demo/generate_demo.py`

- `save_sample_output(output_text)`
  - Saves generated text into `results/gpt_sample_output.txt`.

- `run_generation_demo()`
  - Trains briefly in memory.
  - Starts with the prompt `ai`.
  - Generates short text.

- `main()`
  - Runs the generation demo.

## Sample Outputs

### Trading Sample

```text
=== Mock KIS API Trading System Demo ===
Mode: MOCK
Dry run: True
Access token concept: mock_access_token
Stock symbol: 005930
Current price: 71,500 KRW
Moving average: 73,300.00 KRW
Generated signal: BUY
Explanation: Current price is below the moving average by more than the threshold, so the mock strategy prints BUY.
Order status: No real order was placed. This is a simulation only.
```

### GPT Sample

```text
=== Mini GPT-2 Text Generation Demo ===
Block size: 32
Prompt: ai
Generated text: aikel ol nd bud lear gpt. caler a morar padet bud. mo. ol lder ar gp han od fl olk
```

The generated text is not perfect English because the model is intentionally tiny. The purpose is to demonstrate the GPT structure.

## Safety Notes

- No real API keys are included.
- No real app secrets are included.
- No real account numbers are included.
- `.env` is ignored by `.gitignore`.
- Only `.env.example` is included, and it contains fake placeholders.
- Mock mode is enabled by default.
- Dry-run mode is enabled by default.
- Real trading is disabled by default.
- The program prints simulated signals only.
- The real KIS API functions are templates only.

## Oral Test Preparation

### 1-Minute Project Introduction

English:

> This project has two parts. The first part is a mock automatic trading system based on the concept of the Korea Investment Securities API. It demonstrates authentication, access token usage, market price requests, and a simple moving-average trading signal. It runs in mock mode, so it does not need real API keys and does not place real orders. The second part is a mini GPT-2 demo using PyTorch. It includes token embeddings, positional embeddings, masked multi-head self-attention, transformer blocks, and simple text generation. The goal is not production performance, but understanding the core ideas of AI and financial engineering.

Korean:

> 이 프로젝트는 두 부분으로 구성되어 있습니다. 첫 번째는 한국투자증권 API 개념을 활용한 모의 자동매매 시스템입니다. 인증, access token, 주가 요청, 이동평균 기반 매매 신호를 보여 줍니다. 기본 설정은 mock mode이기 때문에 실제 API 키가 필요 없고 실제 주문도 실행하지 않습니다. 두 번째는 PyTorch로 만든 작은 GPT-2 구조 데모입니다. token embedding, positional embedding, masked multi-head self-attention, transformer block, text generation을 포함합니다. 목적은 고성능 모델 제작이 아니라 핵심 개념을 이해하고 설명하는 것입니다.

Chinese:

> 这个项目分成两部分。第一部分是一个模拟自动交易系统，用来展示韩国投资证券 KIS API 的概念，包括认证、access token、行情请求和移动平均线交易信号。默认是 mock mode，所以不需要真实 API key，也不会下真实订单。第二部分是一个很小的 GPT-2 风格模型，用 PyTorch 展示 embedding、masked self-attention、transformer block 和文本生成。重点是理解和讲解原理，不是做高性能系统。

### Likely Professor Questions And Simple Answers

#### Q1. What is mock mode?

Korean answer:

> Mock mode는 실제 API에 접속하지 않고 가짜 데이터를 사용하는 모드입니다. 그래서 API 키 없이도 프로그램을 실행할 수 있고 안전합니다.

Chinese explanation:

> Mock mode 就是不连接真实 API，只用假数据运行，所以安全、容易演示。

#### Q2. What is dry-run mode?

Korean answer:

> Dry-run mode는 실제 주문을 넣지 않고, 주문을 넣는다면 어떤 결과가 나올지 출력만 하는 방식입니다.

Chinese explanation:

> Dry-run mode 表示只打印模拟结果，不会真的买卖股票。

#### Q3. How does the trading strategy decide BUY or SELL?

Korean answer:

> 현재 가격이 이동평균보다 2% 이상 낮으면 BUY, 2% 이상 높으면 SELL, 그 사이이면 HOLD를 출력합니다.

Chinese explanation:

> 当前价格比移动平均低超过 2% 就 BUY，高超过 2% 就 SELL，否则 HOLD。

#### Q4. What is an access token?

Korean answer:

> Access token은 API 서버가 클라이언트에게 주는 임시 인증 정보입니다. 이후 요청에서 이 토큰을 사용해 권한을 증명합니다.

Chinese explanation:

> Access token 就像临时通行证，证明程序有权限访问 API。

#### Q5. Why is masked self-attention needed in GPT?

Korean answer:

> GPT는 다음 토큰을 예측해야 하므로 미래 토큰을 보면 안 됩니다. Masked self-attention은 미래 위치를 가려서 이전 정보만 보게 합니다.

Chinese explanation:

> GPT 预测下一个 token，不能偷看未来，所以需要 mask 把后面的 token 遮住。

#### Q6. What are Q, K, and V?

Korean answer:

> Q는 query로 현재 위치가 찾는 정보이고, K는 key로 각 위치가 가진 특징이며, V는 value로 실제로 전달되는 정보입니다.

Chinese explanation:

> Q 是查询，K 是匹配用的键，V 是真正被加权传递的信息。

#### Q7. Why is this model not a full GPT-2?

Korean answer:

> 이 모델은 교육용으로 만든 매우 작은 character-level transformer입니다. GPT-2의 핵심 구조는 보여 주지만, 실제 GPT-2처럼 큰 데이터와 큰 모델을 사용하지 않습니다.

Chinese explanation:

> 这个模型只是教学版，结构像 GPT，但是规模很小，用字符训练，不是完整的大型 GPT-2。

## Public GitHub Safety Check

This repository is safe to upload publicly because:

- It contains no real secrets.
- `.env` is ignored.
- `.env.example` uses fake values only.
- Mock mode is the default.
- Real trading is disabled.
- Real API functions are templates only.
