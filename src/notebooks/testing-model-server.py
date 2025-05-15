# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-pycharm
#     notebook_metadata_filter: all,-pycharm
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 2
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython2
#     version: 2.7.6
# ---

# %% [markdown]
# ## Testing calls to model server

# %% [markdown]
# #### Setup for Seldom MLflow custom model server

# %%
# Config for diabetes-model-server
# %env MODEL_SERVER_URL=http://localhost:7000
# %env MODEL_NAME=mlflow-model



# %% language="bash"
#
# docker-compose up -d diabetes-model-server
#

# %% [markdown]
# ### Setup for Seldom mlserver

# %%
# Config for mlserver (Seldon.IO container)
# %env MODEL_SERVER_URL=http://localhost:8080
# %env MODEL_NAME=diabetes-model

# %% language="bash"
#
# docker-compose up -d mlserver
#

# %% [markdown]
# ### Calls to server

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.099537Z", "start_time": "2025-05-15T11:15:23.827935Z"} language="bash"
#
# curl -X POST -s "${MODEL_SERVER_URL}/v2/models/${MODEL_NAME}/infer" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "inputs": [
#              {
#                "name": "input-0",
#                "shape": [2, 10],
#                "datatype": "FP64",
#                "data": [
#                     [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646],
#                     [0.041708, 0.059182, 0.063738, 0.022681, -0.042640, -0.034450, -0.042857, -0.002639, 0.020058, -0.017646]
#                ]
#              }
#            ]
#          }' | jq
#

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.142609Z", "start_time": "2025-05-15T11:15:24.108514Z"} language="bash"
# curl -X GET -s "${MODEL_SERVER_URL}/v2/models/${MODEL_NAME}" | jq
#

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.227694Z", "start_time": "2025-05-15T11:15:24.159253Z"} language="bash"
# curl -s -X POST ${MODEL_SERVER_URL}/invocations \
#  -H "Content-Type: application/json" \
#   -d '{
#         "dataframe_split": {
#             "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
#             "data": [
#                 [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646],
#                 [-0.001882, 0.352598, -0.170647, 0.190409, 0.273711, -0.024980, -0.091299, 0.042257, -0.049390, -0.092804]
#             ]
#         }
#       }' \
#   | jq
#
#

# %% [markdown]
# ##### Force model unload/load with curl

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.299953Z", "start_time": "2025-05-15T11:15:24.240664Z"} language="bash"
#
# curl -s  -X POST "${MODEL_SERVER_URL}/v2/repository/models/${MODEL_NAME}/unload"
#

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.685170Z", "start_time": "2025-05-15T11:15:24.308048Z"} language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/v2/repository/models/${MODEL_NAME}/load" | jq
#

# %% [markdown]
# Check model status with repository index call

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.717244Z", "start_time": "2025-05-15T11:15:24.692559Z"} language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/v2/repository/index" \
#   -H "Content-Type: application/json" \
#   -d "{}" | jq

# %% [markdown]
# #### Using Custom docker container model-server

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.761049Z", "start_time": "2025-05-15T11:15:24.726037Z"} language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/invocations" \
#  -H "Content-Type: application/json" \
#   -d '{
#         "dataframe_split": {
#             "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
#             "data": [
#                 [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646],
#                 [-0.001882, -0.044642, -0.118358, -0.034689, -0.045946, -0.034157, -0.072402, 0.042324, -0.009637, 0.034509]
#             ]
#         }
#       }' \
#   | jq
#

# %% ExecuteTime={"end_time": "2025-05-15T11:15:24.794915Z", "start_time": "2025-05-15T11:15:24.770643Z"} language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/v2/repository/index" \
#   -H "Content-Type: application/json" \
#   -d "{}" | jq
#
