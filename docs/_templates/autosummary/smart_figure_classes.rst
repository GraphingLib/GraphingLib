{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
        :toctree: ./
    {% for item in methods %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
        ~{{ name }}.__iter__
        ~{{ name }}.__getitem__
    {%- if name not in ['SmartTwinAxis'] %}
        ~{{ name }}.__setitem__
        ~{{ name }}.show_grid
        ~{{ name }}.hide_custom_legend_elements
        ~{{ name }}.hide_default_legend_elements
        ~{{ name }}.is_single_subplot
    {%- endif %}
    {% endif %}
    {% endblock %}
