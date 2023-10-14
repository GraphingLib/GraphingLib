from os import listdir
from os.path import dirname

from platformdirs import user_config_dir

from graphinglib.file_manager import FileDeleter, FileLoader, FileSaver


def main_cli():
    """
    The main function for the command line interface.
    """
    # Ask the user if they want to create a new style, edit an existing style, or delete an existing style
    print("\n" + "=" * 38)
    print("Welcome to GraphingLib's style editor!")
    print("=" * 38 + "\n")
    print("What would you like to do?")
    print("1. Create a new style")
    print("2. Edit an existing style")
    print("3. Delete an existing style")
    print("4. Exit")
    print("\n")

    # Ask the user for their choice and validate it (must be 1, 2, 3, or 4)

    choice = input("Enter a number: ").strip()
    while choice not in ["1", "2", "3", "4"]:
        choice = input("You must enter 1, 2, 3, or 4. Enter a number: ").strip()

    print("\n")
    # If the user chose 1, create a new style
    if choice == "1":
        create_new_style()
    # If the user chose 2, edit an existing style
    if choice == "2":
        modify_style()
    # If the user chose 3, delete an existing style
    if choice == "3":
        delete_style()
    # If the user chose 4, exit the program
    if choice == "4":
        print("Goodbye!")


def create_new_style():
    print("=" * 20)
    print("Creating a new style")
    print("=" * 20 + "\n")
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
    print("\nChoose an existing style to base your new style on:")
    for i, name in enumerate(existing_style_names):
        print(f"{i+1}. {name}")
    base_style_index = int(input("\nEnter a number: ").strip()) - 1
    base_style_name = existing_style_names[base_style_index]

    base_style_file = FileLoader(base_style_name)
    default_preferences = base_style_file.load()

    user_preferences = {}
    for section_name, default_values in default_preferences.items():
        print("\n")
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
    print("\n")
    file_saver = FileSaver(filename, user_preferences)
    file_saver.save()

    print(
        f"You can now use your new style by calling gl.Figure(figure_style='{filename}')\n"
    )


def modify_style():
    """
    Modify an existing style.
    """
    print("=" * 27)
    print("Modifying an existing style")
    print("=" * 27 + "\n")

    # Get the names of all the styles
    all_styles = get_style_names(default=True, user=True)

    # Ask the user which style they want to modify
    print("Which style do you want to modify?")
    for i, style_name in enumerate(all_styles):
        print(f"{i+1}. {style_name}")
    choice = input("\nEnter a number: ").strip()

    # Validate the user's choice
    while not choice.isdigit() or int(choice) not in range(1, len(all_styles) + 1):
        choice = input(
            f"You must enter a number between 1 and {len(all_styles)}. Enter a number: "
        ).strip()
    choice = int(choice)

    # Modify the style
    style_name = all_styles[choice - 1]
    file_loader = FileLoader(style_name)
    default_preferences = file_loader.load()

    user_preferences = {}
    for section_name, default_values in default_preferences.items():
        print("\n")
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
    file_saver = FileSaver(style_name, user_preferences)
    file_saver.save()


def delete_style():
    """
    Delete an existing style.
    """
    print("=" * 27)
    print("Deleting an existing style")
    print("=" * 27 + "\n")
    # Get the names of all the styles
    all_styles = get_style_names(default=False, user=True)

    if not all_styles:
        print("There are no styles to delete. Aborting...")
        return

    # Ask the user which style they want to delete
    print("Which style do you want to delete?")
    for i, style_name in enumerate(all_styles):
        print(f"{i+1}. {style_name}")
    choice = input("\nEnter a number: ").strip()

    # Validate the user's choice
    while not choice.isdigit() or int(choice) not in range(1, len(all_styles) + 1):
        choice = input(
            f"You must enter a number between 1 and {len(all_styles)}. Enter a number: "
        ).strip()
    choice = int(choice)

    # Confirm the user's choice
    print(f"\nAre you sure you want to delete {all_styles[choice-1]}?")
    confirm = input("(y/n): ").strip().lower()
    while confirm not in ["yes", "y", "no", "n"]:
        confirm = input("You must enter yes or no: ").strip().lower()
    if confirm in ["no", "n"]:
        print("\nAborting...")
    elif confirm in ["yes", "y"]:
        # Delete the style
        print("\n")
        style_name = all_styles[choice - 1]
        file_saver = FileDeleter(style_name)
        file_saver.delete()


def prompt_for_section(section_name, default_values):
    user_values = {}

    print(f"\n{'-'*30}\n{section_name} settings:\n{'-'*30}")
    for key, default_value in default_values.items():
        user_input = input(f"{key} (default: {default_value}): ").strip()

        # validate the user input and reask if type is wrong
        while True:
            if not user_input:
                user_input = default_value
            else:
                if isinstance(default_value, bool):
                    try:
                        assert user_input.lower() in ["true", "t", "false", "f"]
                        if user_input.lower() in ["true", "t"]:
                            user_input = True
                        elif user_input.lower() in ["false", "f"]:
                            user_input = False
                    except AssertionError:
                        user_input = None
                elif isinstance(default_value, int) or isinstance(default_value, float):
                    try:
                        user_input_int = int(user_input)
                    except ValueError:
                        user_input_int = None

                    try:
                        user_input_float = float(user_input)
                    except ValueError:
                        user_input_float = None

                    if user_input_int is not None:
                        user_input = user_input_int
                    elif user_input_float is not None:
                        user_input = user_input_float
                    else:
                        user_input = None
                elif isinstance(default_value, list):
                    try:
                        assert len(user_input.split(",")) == len(default_value)
                        user_input = [
                            float(i) if "." in i else int(i)
                            for i in user_input.split(",")
                        ]
                    except (ValueError, AssertionError):
                        user_input = None
                elif default_value.startswith("same as"):
                    user_input = user_input
                elif isinstance(default_value, str):
                    try:
                        user_input = float(user_input)
                        user_input = None
                    except ValueError:
                        user_input = user_input

            if user_input is None:
                user_input = input(
                    f"Invalid input for {key}. Please enter a {type(default_value).__name__} or use default value {default_value}: "
                ).strip()
            else:
                user_values[key] = user_input
                break
    return user_values


def get_style_names(user=True, default=True) -> list[str]:
    """
    Get the names of all the styles in the graphinglib.
    """
    all_styles = []
    if default:
        # get the names of all the styles in the default_styles folder
        default_styles_path = f"{dirname(__file__)}/default_styles"
        default_styles = listdir(default_styles_path)
        default_styles = [
            style_name.replace(".yml", "")
            for style_name in default_styles
            if ".yml" in style_name
        ]
        all_styles += default_styles

    if user:
        # add the names of all the styles in the user's config directory
        config_dir = user_config_dir(
            appname="GraphingLib", roaming=True, ensure_exists=True
        )
        user_styles = listdir(config_dir)
        user_styles = [
            style_name.replace(".yml", "")
            for style_name in user_styles
            if ".yml" in style_name
        ]
        all_styles += user_styles

    # combine the two lists and clean it up
    all_styles = list(dict.fromkeys(all_styles))
    all_styles.sort()

    return all_styles


if __name__ == "__main__":
    main_cli()
