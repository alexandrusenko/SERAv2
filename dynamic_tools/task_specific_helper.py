class TaskSpecificHelper:
    def __init__(self):
        self.name = "task_specific_helper"
        self.description = "A tool to help with writing files and handling weather queries for Moscow."

    def run(self, arguments):
        try:
            if 'write_file' in arguments:
                path = arguments['write_file'].get('path')
                content = arguments['write_file'].get('content')
                if not path or not content:
                    return "Error: Both 'path' and 'content' are required for writing a file."
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(content)
                return f"File written successfully to {path}."
            elif 'weather_query' in arguments:
                if arguments['weather_query'] == "какая погода сегодня в москве?":
                    # Simulated weather data for demonstration purposes
                    return "Сегодня в Москве облачно, температура +15°C."
                else:
                    return "Error: Unsupported weather query."
            else:
                return "Error: Unsupported operation. Please use 'write_file' or 'weather_query'."
        except Exception as e:
            return f"An error occurred: {str(e)}"

def build_tool():
    return TaskSpecificHelper()