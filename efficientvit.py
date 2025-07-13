from gradio_client import Client, handle_file

client = Client("https://evitsam.hanlab.ai/")
result = client.predict(
		param_0=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		param_2=64,
		param_3=0.8,
		param_4=0.85,
		param_5=0.7,
		api_name="/lambda_3"
)
print(result)