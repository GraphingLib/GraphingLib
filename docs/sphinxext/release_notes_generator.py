from graphinglib import __version__
from glob import glob
from os.path import split, join, exists
from collections import defaultdict
from github import Github


RELEASE_NOTES_INDEX_TEMPLATE = """
.. _release_notes:

=============
Release Notes
=============

{toctree}
"""

RELEASE_NOTES_TEMPLATE = """
.. _release_notes_{version1}:

GraphingLib {version2} Release Notes
===================================

"""

TAGS = [
    "new_feature",
    "improvement",
    "compatibility",
    "deprecation",
    "expired",
    "change",
]

TAGS_TO_SECTIONS = {
    "new_feature": "New Features",
    "improvement": "Improvements",
    "compatibility": "Compatibility Changes",
    "deprecation": "Deprecations",
    "expired": "Expired Deprecations",
    "change": "Other Changes",
}


# Function to parse version number and return a tuple of integers
def parse_version(version):
    if "dev" in version:
        return tuple(map(int, version.rstrip(".dev").lstrip("v").split(".")))
    else:
        return tuple(map(int, version.lstrip("v").split(".")))


def order_versions(version_numbers):
    # Parse and sort the version numbers
    sorted_versions = sorted(version_numbers, key=parse_version, reverse=True)

    # Group versions by major and minor versions
    grouped_versions = defaultdict(list)
    for version in sorted_versions:
        major, minor, _ = parse_version(version)
        grouped_versions[(major, minor)].append(version)

    # Create the final nested list
    nested_list = list(grouped_versions.values())

    return nested_list


def fetch_old_versions(path):
    old_rn = glob(join(path, "v*.rst"))
    old_v = []
    for file in old_rn:
        fname = split(file)[1]
        v = fname[:-4]
        old_v.append(v)
    return old_v


def fetch_new_files(path):
    upcoming_rn = glob(join(path, "*.*.rst"))
    sorted_upcoming = {}
    pr_list = []
    for file in upcoming_rn:
        fname = split(file)[1]
        prnbr, tag, _ = fname.split(".")
        sorted_upcoming[tag] = sorted_upcoming.get(tag, [])
        sorted_upcoming[tag].append(prnbr)
        pr_list.append(prnbr)
    pr_list.sort()
    return sorted_upcoming, pr_list


def get_highlights(path):
    highlights_file = join(path, "highlights.rst")
    if exists(highlights_file):
        with open(highlights_file, "r") as f:
            highlights = f.readlines()
            f.close()
        return highlights
    return None


def get_github_info(pr_list):
    g = Github()
    org = g.get_organization("GraphingLib")
    repo = org.get_repo("GraphingLib")
    pr_dict = {}
    contrib_dict = {}
    for pr in pr_list:
        pull = repo.get_pull(int(pr))
        pr_dict[pull.title] = pr
        commits = pull.get_commits()
        for c in commits:
            contrib_dict[c.author.login] = None
    return pr_dict, contrib_dict


class ReleaseNoteGenerator:
    def __init__(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            f.close()
        self.title = lines[0]
        self.contents = lines[2:]


def main(app):
    old_v_path = join(app.builder.srcdir, "release_notes", "old_release_notes")
    upcoming_path = join(app.builder.srcdir, "release_notes", "upcoming_changes")
    target_path = join(app.builder.srcdir, "release_notes")
    upcoming, pr_list = fetch_new_files(upcoming_path)
    output = RELEASE_NOTES_TEMPLATE.format(version1=__version__, version2=__version__)

    highlights = get_highlights(upcoming_path)
    if highlights:
        output += "Highlights\n----------\n\n"
        for line in highlights:
            output += line + "\n"
        output += "\n"
    present_tags = [tag for tag in TAGS if tag in upcoming.keys()]

    for tag in present_tags:
        output += (
            TAGS_TO_SECTIONS[tag] + "\n" + "-" * len(TAGS_TO_SECTIONS[tag]) + "\n\n"
        )
        for prn in upcoming[tag]:
            rn = ReleaseNoteGenerator(join(upcoming_path, f"{prn}.{tag}.rst"))
            pr_url = f"https://github.com/GraphingLib/GraphingLib/pull/{prn}"
            output += rn.title + "^" * (len(rn.title) - 1) + "\n\n"
            for line in rn.contents:
                if line == "\n":
                    continue
                output += line + "\n"
            output += f"\n(`pr-{prn} <{pr_url}>`_)\n\n"

    pr_dict, contrib_dict = get_github_info(pr_list)
    output += (
        "Contributors\n------------\n\n"
        + f"A total of {len(contrib_dict)} people contributed to this release.\n\n"
    )
    for contrib in contrib_dict.keys():
        output += f"* `@{contrib} <https://github.com/{contrib}>`_\n\n"
    output += (
        "Merged Pull Requests\n--------------------\n\n"
        + f"A total of {len(pr_dict)} pull requests were merged for this release.\n\n"
    )
    for title in pr_dict.keys():
        output += f"* `#{pr_dict[title]} <https://github.com/GraphingLib/GraphingLib/pull/{pr_dict[title]}>`_ : {title}\n\n"

    if "dev" in __version__:
        with open(join(target_path, f"v{__version__}.rst"), "w") as f:
            f.write(output)
            f.close()
        all_v = fetch_old_versions(old_v_path)
        all_v.append(f"v{__version__}")
        all_v = order_versions(all_v)
    else:
        if upcoming:
            with open(join(old_v_path, f"v{__version__}.rst"), "w") as f:
                f.write(output)
                f.close()
        all_v = fetch_old_versions(old_v_path)
        all_v = order_versions(all_v)

    toctree = ""
    for versions in all_v:
        major_release = versions[0].split(".")[:2]
        major_release = ".".join(major_release)
        toctree += (
            major_release
            + "\n"
            + "-" * len(major_release)
            + "\n\n"
            + ".. toctree::\n   :maxdepth: 2\n\n"
        )
        for v in versions:
            if "dev" in v:
                toctree += "   " + v.lstrip("v") + f" <{v}>\n"
            else:
                toctree += "   " + v.lstrip("v") + f" <{'old_release_notes/'+v}>\n"
        toctree += "\n"

    output = RELEASE_NOTES_INDEX_TEMPLATE.format(toctree=toctree)
    with open(join(target_path, "index.rst"), "w") as f:
        f.write(output)
        f.close()


def setup(app):
    app.connect("builder-inited", main)
