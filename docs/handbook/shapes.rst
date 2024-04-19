===============
Creating Shapes
===============
The :class:`~graphinglib.shapes.Polygon` Object
-------------------------------------------------

GraphingLib allows you to create polygons by specifying their vertices. Here is an example with a random shape: ::

    import graphinglib as gl

    polygon = gl.Polygon(
        vertices=[(0, 0), (1, 1), (2, 0), (1, -1)],
        edge_color="C0",
        line_width=2,
        line_style="solid",
        fill=True,
        fill_color="C0",
        fill_alpha=0.5,
    )

The only required parameter is ``vertices``, but you can customize the appearance of the polygon by specifying the other parameters above. The real power of the :class:`~graphinglib.shapes.Polygon` object comes from its methods. Here are some methods used to get information about a polygon: ::

    print(polygon.get_area())
    print(polygon.get_perimeter())
    print(polygon.get_centroid_coordinates())

    # Check if a point is inside the polygon
    print(Point(1, 0) in polygon)

    # Get the intersection points of the polygon with a curve, or the intersection points of the edges of two polygons
    curve = gl.Curve(lambda x: x ** 2, x_min=-1, x_max=1)
    print(polygon.get_intersection_coordinates(curve))

With GraphingLib, whenever you see a "get_..._coordinates" method, you can safely assume there also exists a "create_..._point" or "create_..._points" method in order to get :class:`~graphinglib.graph_elements.Point` objects instead of coordinates (and vice versa).

There are also many methods which manipulate and transform polygons. Here are some examples: ::

    # Translate, rotate, scale, and skew
    polygon.translate(dx=1, dy=1)
    polygon.rotate(45) # use_rad parameter can be set to True to use radians
    polygon.scale(x_scale=2, y_scale=2)
    polygon.skew(x_skew=30, y_skew=30) # use_rad exists for skew as well

    # Apply an arbitrary linear transformation matrix to the polygon
    polygon.linear_transformation(matrix=np.array([[1, 2], [3, 4]]))

    # Union, intersection, and difference of polygons
    union = polygon.create_union(polygon2)
    intersection = polygon.create_intersection(polygon2)
    difference = polygon.create_difference(polygon2)

    # Split a polygon into multiple polygons using a Curve
    curve = gl.Curve(lambda x: x ** 2, x_min=-1, x_max=1)
    polygons = polygon.split(curve)

Some of the most common shapes, such as rectangles and circles, have their dedicated classes to simplify their creation. These classes are detailed below.

The :class:`~graphinglib.shapes.Rectangle` Object
-------------------------------------------------

Rectangles can be created easily by creating an instance of the :class:`~graphinglib.shapes.Rectangle` class as shown below: ::

    import graphinglib as gl

    # Create a Rectangle from the bottom left corner
    rect = gl.Rectangle(x_bottom_left=0, y_bottom_left=0, width=10, height=10)

    # Create a Rectangle from its center
    rect2 = gl.Rectangle.from_center(x=0, y=0, width=10, height=10)

You can customize the appearance of Rectangles by specifying the following optional parameters: ``edge_color``, ``line_width``, ``line_style``, ``fill`` (True or False), ``fill_color``, and ``fill_alpha``. Here is an example with different styles of Rectangles: ::

    import graphinglib as gl

    rect1 = gl.Rectangle(
        x_bottom_left=2,
        y_bottom_left=2,
        width=10,
        height=10,
        fill_color="red",
        edge_color="red",
        line_width=1,
        line_style="solid",
        fill=True,
        fill_alpha=1,
    )

    rect2 = gl.Rectangle(
        x_bottom_left=5,
        y_bottom_left=5,
        width=5,
        height=12,
        color="blue",
        line_width=2,
        line_style="dashed",
        fill=True,
        fill_alpha=0.5,
    )

    rect3 = gl.Rectangle(
        x_bottom_left=0,
        y_bottom_left=0,
        width=14,
        height=19,
        color="green",
        line_width=5,
        line_style="dotted",
        fill=False,
    )

    figure = gl.Figure()
    figure.add_elements(rect1, rect2, rect3)
    figure.show()

.. image:: images/rectangle.png

All :class:`~graphinglib.shapes.Polygon` methods can also be used with :class:`~graphinglib.shapes.Rectangle` objects.


The :class:`~graphinglib.shapes.Circle` Object
-----------------------------------------------

GraphingLib also lets you plot Circles. You can create a Circle by specifying its center point and radius: ::

    import graphinglib as gl

    circle = gl.Circle(x_center=0, y_center=0, radius=10)

You can customize the appearance of Circles by specifying the following optional parameters: ``color``, ``line_width``, ``line_style``, ``fill`` (True or False), and ``fill_alpha``. Here is an example with different styles of Circles: ::

    import graphinglib as gl

    circle1 = gl.Circle(
        x_center=-4,
        y_center=6,
        radius=10,
        fill_color="red",
        edge_color="red",
        line_width=1,
        line_style="solid",
        fill=True,
        fill_alpha=1,
    )

    circle2 = gl.Circle(
        x_center=4,
        y_center=6,
        radius=7,
        fill_color="blue",
        edge_color="blue",
        line_width=2,
        line_style="dashed",
        fill=True,
        fill_alpha=0.5,
    )

    circle3 = gl.Circle(
        x_center=0,
        y_center=-4,
        radius=13,
        fill_color="green",
        edge_color="green",
        line_width=5,
        line_style="dotted",
        fill=False,
    )

    # Figure size and axis limits are set to make the circles look round
    figure = gl.Figure(x_lim=(-19, 19), y_lim=(-19, 19), size=(8, 8))
    figure.add_elements(circle1, circle2, circle3)
    figure.show()

.. image:: images/circle.png

As with Rectangles, all :class:`~graphinglib.shapes.Polygon` methods can also be used with :class:`~graphinglib.shapes.Circle` objects.

Since :class:`~graphinglib.shapes.Circle` objects actually inherit from :class:`~graphinglib.shapes.Polygon`, they aren't perfectly round, and so area and perimeter calculations are approximations. You can get arbitrarily close to the true values by increasing the number of points used to approximate the circle. This can be done by setting the ``number_of_points`` parameter when creating the Circle object. The default value is 100, which gives you 99.9% accuracy for the area and even better for the perimeter. Here is an example: ::

    import graphinglib as gl

    circle = gl.Circle(x_center=0, y_center=0, radius=10, number_of_points=1000)
    print(circle.get_area())
    print(circle.get_perimeter())

The :class:`~graphinglib.shapes.Arrow` Object
----------------------------------------------

GraphingLib also lets you plot Arrows. You can create an Arrow by specifying its start and end points: ::

    import graphinglib as gl

    arrow = gl.Arrow(pointA=(0, 0), pointB=(10, 10))

You can customize the appearance of Arrows by specifying the following optional parameters: ``color``, ``width`` (the line width), ``head_size``, ``two_sided`` (True or False), and ``shrink``. The ``shrink`` parameter is a float between 0 and 0.5 which shortens the arrow from both ends by the given percentage (0 doesn't shrink at all, 0.5 makes the arrow disappear completely). Here is an example with different styles of Arrows: ::

    import graphinglib as gl

    arrow1 = gl.Arrow(
        pointA=(0, 0),
        pointB=(1, 1),
        color="red",
        shrink=0,  # default, no shrinking
    )
    arrow2 = gl.Arrow(
        pointA=(1, 0),
        pointB=(2, 1),
        color="blue",
        shrink=0.05,
        two_sided=True,
        head_size=3,
    )
    arrow3 = gl.Arrow(
        pointA=(2, 0),
        pointB=(3, 1),
        color="green",
        shrink=0.2,
        two_sided=True,
        width=4,
    )

    # Create points at the start and end of the arrows (to illustrate the shrinking)
    point1 = gl.Point(0, 0, color="red")
    point2 = gl.Point(1, 0, color="blue")
    point3 = gl.Point(2, 0, color="green")
    point4 = gl.Point(1, 1, color="red")
    point5 = gl.Point(2, 1, color="blue")
    point6 = gl.Point(3, 1, color="green")

    fig = gl.Figure(y_lim=(-0.5, 1.5), x_lim=(-0.5, 3.5))
    fig.add_elements(arrow1, arrow2, arrow3)
    fig.add_elements(point1, point2, point3)
    fig.add_elements(point4, point5, point6)
    fig.show()

.. image:: images/arrow.png

The :class:`~graphinglib.shapes.Line` object
--------------------------------------------

It is possible to add lines to figures. Similarly to the :class:`~graphinglib.shapes.Arrow` object, simply specify the two end points::

    import graphinglib as gl

    line = gl.Line((0, 0), (1, 1))

It is possible to change the width of the line with the ``width`` parameter. The ``capped_line`` parameter allows you to add perpendicular caps to both ends of the line. The width of those caps can be controlled with the ``cap_width`` parameter::

    import graphinglib as gl

    # Creating a circle and finding a point at 45 degrees on the circumference
    circle = gl.Circle(0, 0, 1, line_width=2)
    center = gl.Point(0, 0, marker_size=50)
    point = circle.create_point_at_angle(45, degrees=True)
    point.marker_size = 50
    
    # Adding a line to display the radius of the circle
    line = gl.Line(
        (-0.07, 0.07), (point.x - 0.07, point.y + 0.07), capped_line=True, cap_width=1
    )
    text = gl.Text(0.25, 0.5, r"$R$", font_size=15)

    # Display the elements
    fig = gl.Figure(size=(5.5, 5))
    fig.add_elements(circle, point, line, center, text)
    fig.show()

.. image:: images/capped_line.png