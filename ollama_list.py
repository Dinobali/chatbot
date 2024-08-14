import subprocess
import json

def fetch_available_models():
    # Run the `ollama list` command and capture the output
    try:
        result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout
        print("Raw output from `ollama list`:\n", output)  # Print raw output for debugging
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama list: {e}")
        return []

    # Parse the output to extract model names
    models = []
    excluded_keywords = {'tge', 'bge', 'text', 'embed'}
    
    for line in output.splitlines():
        # Example logic to extract model names; adjust this based on actual output format
        if line.strip() and not line.startswith("NAME"):
            # Assuming models are listed in the second column, adjust as needed
            parts = line.split()
            if parts:
                model_name = parts[0].strip()
                # Check if the model name contains any excluded keywords
                if not any(keyword in model_name for keyword in excluded_keywords):
                    models.append(model_name)
    
    return models

def update_config(models):
    # Read the current config.json
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        config = {}
    except json.JSONDecodeError:
        print("Error reading config.json")
        return

    # Update the available_models in config
    config['available_models'] = models

    # Write the updated config back to config.json
    try:
        with open('config.json', 'w') as file:
            json.dump(config, file, indent=4)
    except IOError as e:
        print(f"Error writing to config.json: {e}")

def main():
    models = fetch_available_models()
    if models:
        update_config(models)
        print("config.json has been updated with the latest models.")
    else:
        print("No models found or there was an error fetching models.")

if __name__ == '__main__':
    main()
