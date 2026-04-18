python eval.py \
  --model_family qwen \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --test_json /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/CultureDrive/CultureDrive_Benchmark_Mini_v1.json \
  --output_json /weka/home/ext-yingzima/CulturalDrive/results/qwen2.5vl_r_output.json \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --is_reasoning True


python eval.py \
  --model_family glm \
  --model_path zai-org/GLM-4.6V-Flash \
  --test_json /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/CultureDrive/CultureDrive_Benchmark_Mini_v1.json \
  --output_json /weka/home/ext-yingzima/CulturalDrive/results/glm4.6v_output.json \
  --max_new_tokens 64 \
  --dtype bfloat16


python eval.py \
  --model_family internvl \
  --model_path OpenGVLab/InternVL3-8B \
  --test_json /weka/home/ext-yingzima/CulturalDrive/traffic_handbook/CultureDrive/CultureDrive_Benchmark_Mini_v1.json \
  --output_json /weka/home/ext-yingzima/CulturalDrive/results/internvl_output.json \
  --max_new_tokens 64 \
  --dtype bfloat16