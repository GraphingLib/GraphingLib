from os import listdir
from os.path import dirname

from platformdirs import user_config_dir

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
    existing_style_names = get_style_names()
    filename = input(
        "Enter a unique name for your custom style (e.g., big_bold_figure) or press Enter for default (style_1): "
    ).strip()

    # check if the filename is already in use
    while filename in existing_style_names:
        filename = input(
            f"The filename {filename} is already in use. Enter a unique name for your custom style (e.g., style_1): "
        ).strip()

    if not filename:
        filename = "style_1"
        while filename in existing_style_names:
            filename = (
                filename.split("_")[0] + "_" + str(int(filename.split("_")[-1]) + 1)
            )

    # ask for style name to be used as a base
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
    default_styles_path = f"{dirname(__file__)}/default_styles"
    default_styles = listdir(default_styles_path)
    default_styles = [
        style_name.replace(".yml", "")
        for style_name in default_styles
        if ".yml" in style_name
    ]

    # add the names of all the styles in the user's config directory
    config_dir = user_config_dir(appname="GraphingLib", roaming=True)
    user_styles = listdir(config_dir)
    user_styles = [
        style_name.replace(".yml", "")
        for style_name in user_styles
        if ".yml" in style_name
    ]

    # combine the two lists and clean it up
    all_styles = default_styles + user_styles
    all_styles = list(dict.fromkeys(all_styles))
    all_styles.sort()

    return all_styles


def main_cli():
    """
    The main function for the command line interface.
    """
    # Ask the user if they want to create a new style, edit an existing style, or delete an existing style
    print("Welcome to GraphingLib's style editor!")
    print("What would you like to do?")
    print("1. Create a new style")
    print("2. Edit an existing style")
    print("3. Delete an existing style")
    print("4. Exit")

    # Ask the user for their choice and validate it (must be 1, 2, 3, or 4)

    choice = input("Enter a number: ").strip()
    while choice not in ["1", "2", "3", "4"]:
        choice = input("You must enter 1, 2, 3, or 4. Enter a number: ").strip()

    # If the user chose 1, create a new style
    if choice == "1":
        create_new_style()
    # If the user chose 2, edit an existing style
    if choice == "2":
        # TODO: implement this
        print("This feature is not yet implemented.")
    # If the user chose 3, delete an existing style
    if choice == "3":
        # TODO: implement this
        print("This feature is not yet implemented.")
    # If the user chose 4, exit the program
    if choice == "4":
        print("Goodbye!")


if __name__ == "__main__":
    main_cli()