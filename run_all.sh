

python3 -m varbench.run_evaluation --model deepseek-r1-distill-llama-70b --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 1 --agent simpleLLM;
python3 -m varbench.run_evaluation --model deepseek-r1-distill-llama-70b --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 5 --agent simpleLLM
python3 -m varbench.run_evaluation --model deepseek-r1-distill-llama-70b --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 1.5 --passk 5 --agent simpleLLM
python3 -m varbench.run_evaluation --model llama-3.1-8b-instant --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 1 --agent simpleLLM;
python3 -m varbench.run_evaluation --model llama-3.3-70b-versatile --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 1 --agent simpleLLM;
python3 -m varbench.run_evaluation --model llama-3.3-70b-versatile --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 5 --agent simpleLLM
python3 -m varbench.run_evaluation --model llama3-70b-8192 --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 1 --agent simpleLLM;
python3 -m varbench.run_evaluation --model mixtral-8x7b-32768 --api_url https://api.groq.com/openai/v1 --api_key $GROQ_API_KEY --temperature 0.7 --passk 1 --agent simpleLLM;
python3 -m varbench.run_evaluation --model gpt-4o-2024-08-06 --api_url https://api.openai.com/v1  --temperature 0.7 --passk 5 --agent simpleLMM
python3 -m varbench.run_evaluation --model gpt-4o-2024-08-06 --api_url https://api.openai.com/v1  --temperature 0.7 --passk 5 --agent simpleLLM

