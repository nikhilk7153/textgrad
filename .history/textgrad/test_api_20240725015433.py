from openai import AzureOpenAI

client = AzureOpenAI(
    api_version = "2024-03-01-preview", 
    azure_endpoint = "https://bionlp-gpt4-wang.openai.azure.com/",
    api_key =  "a494edc84d714b6c8a12e7212974b793"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is pulmonary embolism?"}

]

response = client.chat.completions.create(
    model = "gpt-35-turbo",
    messages = messages, 
    temperature=0

)

print(response.choices[0].message.content)