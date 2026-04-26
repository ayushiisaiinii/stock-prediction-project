{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uvxmastjP0Bq",
        "outputId": "1d820d08-df5e-4864-c52d-04dd6e589f0d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: yfinance in /usr/local/lib/python3.12/dist-packages (0.2.66)\n",
            "Requirement already satisfied: pandas>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from yfinance) (2.2.2)\n",
            "Requirement already satisfied: numpy>=1.16.5 in /usr/local/lib/python3.12/dist-packages (from yfinance) (2.0.2)\n",
            "Requirement already satisfied: requests>=2.31 in /usr/local/lib/python3.12/dist-packages (from yfinance) (2.32.4)\n",
            "Requirement already satisfied: multitasking>=0.0.7 in /usr/local/lib/python3.12/dist-packages (from yfinance) (0.0.12)\n",
            "Requirement already satisfied: platformdirs>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from yfinance) (4.9.4)\n",
            "Requirement already satisfied: pytz>=2022.5 in /usr/local/lib/python3.12/dist-packages (from yfinance) (2025.2)\n",
            "Requirement already satisfied: frozendict>=2.3.4 in /usr/local/lib/python3.12/dist-packages (from yfinance) (2.4.7)\n",
            "Requirement already satisfied: peewee>=3.16.2 in /usr/local/lib/python3.12/dist-packages (from yfinance) (4.0.3)\n",
            "Requirement already satisfied: beautifulsoup4>=4.11.1 in /usr/local/lib/python3.12/dist-packages (from yfinance) (4.13.5)\n",
            "Requirement already satisfied: curl_cffi>=0.7 in /usr/local/lib/python3.12/dist-packages (from yfinance) (0.14.0)\n",
            "Requirement already satisfied: protobuf>=3.19.0 in /usr/local/lib/python3.12/dist-packages (from yfinance) (5.29.6)\n",
            "Requirement already satisfied: websockets>=13.0 in /usr/local/lib/python3.12/dist-packages (from yfinance) (15.0.1)\n",
            "Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.12/dist-packages (from beautifulsoup4>=4.11.1->yfinance) (2.8.3)\n",
            "Requirement already satisfied: typing-extensions>=4.0.0 in /usr/local/lib/python3.12/dist-packages (from beautifulsoup4>=4.11.1->yfinance) (4.15.0)\n",
            "Requirement already satisfied: cffi>=1.12.0 in /usr/local/lib/python3.12/dist-packages (from curl_cffi>=0.7->yfinance) (2.0.0)\n",
            "Requirement already satisfied: certifi>=2024.2.2 in /usr/local/lib/python3.12/dist-packages (from curl_cffi>=0.7->yfinance) (2026.2.25)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas>=1.3.0->yfinance) (2.9.0.post0)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas>=1.3.0->yfinance) (2025.3)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests>=2.31->yfinance) (3.4.6)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests>=2.31->yfinance) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests>=2.31->yfinance) (2.5.0)\n",
            "Requirement already satisfied: pycparser in /usr/local/lib/python3.12/dist-packages (from cffi>=1.12.0->curl_cffi>=0.7->yfinance) (3.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas>=1.3.0->yfinance) (1.17.0)\n"
          ]
        }
      ],
      "source": [
        "# Install (if needed)\n",
        "!pip install yfinance\n",
        "\n",
        "# Imports\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import yfinance as yf\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import LSTM, Dense"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_stock = yf.download(\"AAPL\", start=\"2015-01-01\", end=\"2024-01-01\")\n",
        "\n",
        "# Flatten columns (important)\n",
        "df_stock.columns = df_stock.columns.get_level_values(0)\n",
        "\n",
        "# Keep Close\n",
        "df_stock = df_stock[['Close']]\n",
        "\n",
        "# Convert index → column\n",
        "df_stock.reset_index(inplace=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ogUcMXdmQE0o",
        "outputId": "cb605167-ec8e-469c-a5a0-5082bfe731c3"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipykernel_9271/1923907414.py:1: FutureWarning: YF.download() has changed argument auto_adjust default to True\n",
            "  df_stock = yf.download(\"AAPL\", start=\"2015-01-01\", end=\"2024-01-01\")\n",
            "\r[*********************100%***********************]  1 of 1 completed\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(df_stock.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VKuZw6NwxwHY",
        "outputId": "aae7c362-5874-4a5c-deec-a694cdc47bad"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['Date', 'Close'], dtype='object', name='Price')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_stock['Date'] = df_stock['Date'].astype(str)"
      ],
      "metadata": {
        "id": "NPid2yt7x4OQ"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_stock.columns = df_stock.columns.get_level_values(0)"
      ],
      "metadata": {
        "id": "-Y67FLIRxTfb"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.upload()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 212
        },
        "id": "apJ_kSk7wKIH",
        "outputId": "bce95e2e-98ec-450a-85c7-6b5a890f4853"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-ce78724f-8417-4f3c-9412-91fb93f63e91\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-ce78724f-8417-4f3c-9412-91fb93f63e91\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving daily_sentiment.csv to daily_sentiment (2).csv\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'daily_sentiment (2).csv': b'date,sentiment\\n2026-03-02,0.0\\n2026-03-03,-0.11026824177980109\\n2026-03-04,0.0\\n2026-03-05,-0.27130230468658967\\n2026-03-06,0.342345392478557\\n2026-03-08,0.0\\n2026-03-09,0.397450692475601\\n2026-03-11,-0.4718946005252107\\n2026-03-13,0.32183846996532267\\n2026-03-15,0.0\\n2026-03-16,0.20350801221239995\\n2026-03-17,0.0\\n2026-03-18,0.4718946005252107\\n2026-03-19,-0.37946068450596504\\n2026-03-20,0.0\\n2026-03-21,0.0\\n2026-03-23,0.3532475557251745\\n2026-03-24,0.28686069102341155\\n2026-03-25,0.0\\n2026-03-26,0.4059768605376926\\n2026-03-27,0.4871796194098732\\n2026-03-29,0.6568262682986041\\n2026-03-30,0.32183846996532267\\n2026-03-31,-0.020286107484387705\\n'}"
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_sent = pd.read_csv(\"daily_sentiment.csv\")"
      ],
      "metadata": {
        "id": "NmbTwtApqQgw"
      },
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.merge(df_stock, df_sent, left_on='Date', right_on='date', how='left')\n",
        "\n",
        "df = df.drop(columns=['date'])\n",
        "df = df.copy()"
      ],
      "metadata": {
        "id": "-xsOyb4Vwet0"
      },
      "execution_count": 50,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df['sentiment'] = df['sentiment'].fillna(0)\n",
        "\n",
        "# Create indicators\n",
        "df['MA10'] = df['Close'].rolling(window=10).mean()\n",
        "df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()\n",
        "\n",
        "delta = df['Close'].diff()\n",
        "gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()\n",
        "loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()\n",
        "\n",
        "rs = gain / loss\n",
        "df['RSI'] = 100 - (100 / (1 + rs))"
      ],
      "metadata": {
        "id": "klS-6BAYrek-"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Smooth sentiment\n",
        "df['sentiment_3d'] = df['sentiment'].rolling(3).mean()\n",
        "df['sentiment_7d'] = df['sentiment'].rolling(7).mean()\n",
        "\n",
        "# Lag features\n",
        "df['sentiment_lag1'] = df['sentiment'].shift(1)\n",
        "df['sentiment_lag2'] = df['sentiment'].shift(2)\n",
        "df['sentiment_lag3'] = df['sentiment'].shift(3)"
      ],
      "metadata": {
        "id": "B05TlyTl_guB"
      },
      "execution_count": 52,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Fill rolling sentiment\n",
        "df['sentiment_3d'] = df['sentiment_3d'].bfill()\n",
        "df['sentiment_7d'] = df['sentiment_7d'].bfill()\n",
        "\n",
        "# Fill lag features\n",
        "df[['sentiment_lag1','sentiment_lag2','sentiment_lag3']] = \\\n",
        "df[['sentiment_lag1','sentiment_lag2','sentiment_lag3']].bfill()\n",
        "\n",
        "# Drop only indicator NaNs\n",
        "df = df.dropna(subset=['MA10', 'EMA10', 'RSI'])\n",
        "\n",
        "df.reset_index(drop=True, inplace=True)"
      ],
      "metadata": {
        "id": "d2koCWMo_jPq"
      },
      "execution_count": 53,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_model = df.drop(columns=['Date'])\n",
        "\n",
        "scaler = MinMaxScaler(feature_range=(0,1))\n",
        "scaled_data = scaler.fit_transform(df_model)"
      ],
      "metadata": {
        "id": "4Z11-GDv_mGt"
      },
      "execution_count": 54,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_size = int(len(scaled_data) * 0.8)\n",
        "\n",
        "train_data = scaled_data[:train_size]\n",
        "test_data = scaled_data[train_size-60:]"
      ],
      "metadata": {
        "id": "SunYQUA7_o3a"
      },
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, y_train = [], []\n",
        "\n",
        "for i in range(60, len(train_data)):\n",
        "    X_train.append(train_data[i-60:i])\n",
        "    y_train.append(train_data[i, 0])\n",
        "\n",
        "X_train, y_train = np.array(X_train), np.array(y_train)\n",
        "\n",
        "\n",
        "X_test, y_test = [], []\n",
        "\n",
        "for i in range(60, len(test_data)):\n",
        "    X_test.append(test_data[i-60:i])\n",
        "    y_test.append(test_data[i, 0])\n",
        "\n",
        "X_test, y_test = np.array(X_test), np.array(y_test)"
      ],
      "metadata": {
        "id": "4ngrHPhB_rWA"
      },
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential()\n",
        "\n",
        "model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))\n",
        "model.add(LSTM(64))\n",
        "model.add(Dense(1))\n",
        "\n",
        "model.compile(optimizer='adam', loss='mean_squared_error')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RV7I1wm3_uAt",
        "outputId": "5afeb2a7-4c3b-46f0-9192-acb7014426c4"
      },
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/keras/src/layers/rnn/rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(**kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(X_train, y_train, epochs=10, batch_size=32)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Y1xo3l-x_xF3",
        "outputId": "4e7b0af2-6bc4-456a-83d2-acf09de9be7b"
      },
      "execution_count": 58,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 51ms/step - loss: 0.0067\n",
            "Epoch 2/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 51ms/step - loss: 4.0888e-04\n",
            "Epoch 3/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 67ms/step - loss: 2.9990e-04\n",
            "Epoch 4/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 51ms/step - loss: 2.9508e-04\n",
            "Epoch 5/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 50ms/step - loss: 2.8071e-04\n",
            "Epoch 6/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 51ms/step - loss: 2.6135e-04\n",
            "Epoch 7/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 72ms/step - loss: 2.4637e-04\n",
            "Epoch 8/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 51ms/step - loss: 2.7412e-04\n",
            "Epoch 9/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 52ms/step - loss: 2.3837e-04\n",
            "Epoch 10/10\n",
            "\u001b[1m55/55\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 52ms/step - loss: 2.3813e-04\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.history.History at 0x7bf7a005ccb0>"
            ]
          },
          "metadata": {},
          "execution_count": 58
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "predictions = model.predict(X_test)\n",
        "\n",
        "dummy = np.zeros((predictions.shape[0], df_model.shape[1]))\n",
        "dummy[:,0] = predictions[:,0]\n",
        "\n",
        "predictions = scaler.inverse_transform(dummy)[:,0]\n",
        "\n",
        "dummy2 = np.zeros((y_test.shape[0], df_model.shape[1]))\n",
        "dummy2[:,0] = y_test\n",
        "\n",
        "y_test_actual = scaler.inverse_transform(dummy2)[:,0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "33Z4mqV7_-7e",
        "outputId": "d5842340-582f-4b84-da9a-0122e3882c3a"
      },
      "execution_count": 59,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m15/15\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 35ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(predictions[:10])\n",
        "print(len(predictions))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2GYiyJS_AQwO",
        "outputId": "832acda5-3e8c-43e7-9ac5-81fbf5b4e312"
      },
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[154.18189106 154.24699359 154.37270088 154.80542769 155.51408714\n",
            " 156.36127774 157.34158126 158.78762309 160.58154087 162.54419805]\n",
            "451\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(y_test_actual, label=\"Actual\")\n",
        "plt.plot(predictions, label=\"Predicted\")\n",
        "plt.legend()\n",
        "plt.title(\"Final Model (Advanced Sentiment FIXED)\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 452
        },
        "id": "lFnKszZDABvc",
        "outputId": "7863e483-f085-4e43-c797-86e89c3af808"
      },
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAigAAAGzCAYAAAAFROyYAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA0c5JREFUeJzsnXd4FVX+h9+5Pb2HECCU0DsiIjZQUcCKuquuDfvaC+qu/nbXuoqufV0su7r2tihix4ICdgUJIL0kJJDe+63z++PcmVsT0ut5nyfPvTNzZubckjuf+VZFVVUViUQikUgkkh6EobsnIJFIJBKJRBKMFCgSiUQikUh6HFKgSCQSiUQi6XFIgSKRSCQSiaTHIQWKRCKRSCSSHocUKBKJRCKRSHocUqBIJBKJRCLpcUiBIpFIJBKJpMchBYpEIpFIJJIehxQokrDk5OSgKAovvfRSp55n2LBhXHzxxZ16jvZw8cUXM2zYsDbtO2fOHObMmdOisbW1taSmpvL666+36VxNcffdd6MoSocesyfTVd/bzqK/fV69ha1bt2Iymfjtt9+6eyr9CilQ+ikvvfQSiqKE/bv99tu7e3ohaHO7/PLLw27/y1/+oo8pLS3t4tm1nyeffJKYmBjOPffcsNv/9Kc/oSgK55xzThfPrO+Sk5PDJZdcQmZmJjabjbS0NI455hjuuuuuTj1vfX09d999N6tXr+7U83QmDzzwACtWrGjRWE00hvs7/PDD9XEXX3wx0dHR+nJxcTGJiYkcd9xxIcd0Op1MmjSJYcOGUVdXBzT/m6YoCj/++KO+v/96k8lEYmIi06dP58Ybb2Tr1q0h5xs/fjwnn3wyd955Z0vfIkkHYOruCUi6l3vvvZfhw4cHrJs4cSJDhw6loaEBs9ncTTMLxWaz8e677/L0009jsVgCtr355pvYbDYaGxu7aXZtx+l08uSTT3LzzTdjNBpDtquqyptvvsmwYcP48MMPqampISYmphtm2nfYvXs3M2bMICIigksvvZRhw4ZRUFDAr7/+ykMPPcQ999zTaeeur6/Xjx9sYfvrX//aI28QgnnggQf43e9+x8KFC1u8zx/+8AdOOumkgHUpKSlNjk9NTeWhhx7iyiuv5OWXX2bRokX6tkcffZTffvuNDz/8kKioqID9wv2mAYwcOTJg+YQTTuCiiy5CVVWqqqrYuHEjL7/8Mk8//TQPPfQQixcvDhh/1VVXcdJJJ7Fnzx4yMzNb/LolbUcKlH7OggULOPTQQ8Nus9lsXTyb5pk/fz4ffPABn376Kaeffrq+/vvvvyc7O5uzzjqLd999txtn2DY++ugjSkpKOPvss8NuX716Nfv37+err75i3rx5LF++PODHWtJ6Hn/8cWpra8nKymLo0KEB24qLi7tpVmAymTCZ+ubP8iGHHMIFF1zQqn0uv/xyXnnlFW699VZOOeUUkpKSyM7O5t577+XMM8/klFNOCdmnud80f0aPHh0ynwcffJBTTz2VW265hbFjxwYIqrlz55KQkMDLL7/Mvffe26rXIWkb0sUjCUs4X75mgj1w4AALFy4kOjqalJQUbr31Vtxud8D+jzzyCEcccQRJSUlEREQwffp03nnnnXbNadCgQRxzzDG88cYbAetff/11Jk2axMSJE8Put2zZMqZPn05ERATJyclccMEFHDhwIGTcihUrmDhxIjabjYkTJ/Lee++FPZ7H4+GJJ55gwoQJ2Gw2BgwYwB//+EcqKira9LpWrFjBsGHDmrwre/311xk/fjzHHnssc+fObTJO5dtvv2XGjBnYbDYyMzN57rnnQsZMnDiRY489NuxrGjRoEL/73e/0dS39DBVF4brrrtPfP6vVyoQJE1i5cmXI2AMHDnDZZZeRnp6O1Wpl+PDhXH311TgcDn1MZWUlN910E0OGDMFqtTJy5EgeeughPB5PwLEqKyu5+OKLiYuLIz4+nkWLFlFZWRn2vQlmz549DB48OEScgLhzD+bTTz/l6KOPJioqipiYGE4++WS2bNkSMKYl/x85OTm61eCee+7R3Qx33303ED4GRXt/ly1bxvjx44mIiGDWrFls3rwZgOeee46RI0dis9mYM2cOOTk5IfP/6aefmD9/PnFxcURGRjJ79my+++67gDHauXfv3s3FF19MfHw8cXFxXHLJJdTX1wfMp66ujpdfflmff2fFkSmKwrPPPktVVRW33norANdccw0mk4l//vOfHX6+pKQk3nrrLUwmE/fff3/ANrPZzJw5c3j//fc7/LyS8PRNqS5pMVVVVSExG8nJyU2Od7vdzJs3j5kzZ/LII4/w5Zdf8uijj5KZmcnVV1+tj3vyySc57bTTOP/883E4HLz11lv8/ve/56OPPuLkk09u83zPO+88brzxRmpra4mOjsbl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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save(\"final_stock_sentiment_model.keras\")"
      ],
      "metadata": {
        "id": "os3F5XxnCz-i"
      },
      "execution_count": 64,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.download(\"final_stock_sentiment_model.keras\")"
      ],
      "metadata": {
        "id": "Hk5mVrkcDOS8",
        "outputId": "9398692d-1fb8-454b-a769-b58e9c66a8a9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17
        }
      },
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_3a750726-0110-408c-923b-80f2f3c50af3\", \"final_stock_sentiment_model.keras\", 659832)"
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}