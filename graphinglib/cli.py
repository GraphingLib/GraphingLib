from graphinglib.file_manager import FileLoader, FileSaver


def prompt_for_section(section_name, default_values):
    user_values = {}

    print(f"{section_name} settings:")
    for key, default_value in default_values.items():
        user_input = input(f"{key} (default: {default_value}): ").strip()

        if "same as" in str(default_value):
            ref_key = default_value.split(" ")[-1]
            default_value = user_values.get(ref_key, default_value)

        if not user_input:
            user_input = default_value

        if isinstance(default_value, bool):
            if not isinstance(user_input, bool):
                user_input = user_input.lower() in ["true", "yes", "y", "1"]
        elif isinstance(default_value, int):
            user_input = int(user_input)
        elif isinstance(default_value, float):
            user_input = float(user_input)
        elif isinstance(default_value, list):
            user_input = [
                float(i) if "." in i else int(i) for i in user_input.split(",")
            ]

        user_values[key] = user_input

    return user_values


def create_new_style():
    filename = input(
        "Enter a name for your custom style (e.g., my_style) or press Enter for default (user_preferences): "
    ).strip()
    if not filename:
        filename = "user_preferences"

    # ask for style name to be used as a base
    existing_style_names = get_style_names()
    print("Choose an existing style to base your new style on:")
    for i, name in enumerate(existing_style_names):
        print(f"{i+1}. {name}")
    base_style_index = int(input("Enter a number: ").strip()) - 1
    base_style_name = existing_style_names[base_style_index]

    base_style_file = FileLoader(base_style_name)
    default_preferences = base_style_file.load()

    user_preferences = {}
    for section_name, default_values in default_preferences.items():
        customize = (
            input(f"Do you want to customize {section_name} settings? (y/n): ")
            .strip()
            .lower()
        )
        if customize in ["yes", "y"]:
            user_values = prompt_for_section(section_name, default_values)
            user_preferences[section_name] = user_values
        else:
            user_preferences[section_name] = default_values

    # Save user preferences to the specified filename
    file_saver = FileSaver(filename, user_preferences)
    file_saver.save()


def get_style_names() -> list[str]:
    """
    Get the names of all the styles in the graphinglib.
    """
    # get the names of all the styles in the default_styles folder
    from os import listdir
    from os.path import dirname

    default_styles_path = f"{dirname(__file__)}/default_styles"
    default_styles = listdir(default_styles_path)
    default_styles = [style_name.replace(".yml", "") for style_name in default_styles]
    return default_styles


create_new_style()
