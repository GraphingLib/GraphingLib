===============
Creating Shapes
===============

The :class:`~graphinglib.shapes.Rectangle` Object
-------------------------------------------------

You can draw rectangles easily by creating an instance of the :class:`~graphinglib.shapes.Rectangle` class as shown below: ::

    import graphinglib as gl

    # Create a Rectangle from the bottom left corner
    rect = gl.Rectangle(x_bottom_left=0, y_bottom_left=0, width=10, height=10)

    # Create a Rectangle from its center
    rect2 = gl.Rectangle.from_center(x=0, y=0, width=10, height=10)

    # Create a Rectangle from two opposite corner points
    point1 = gl.Point(0,0)
    point2 = gl.Point(10,10)
    rect3 = gl.Rectangle.from_points(point1, point2)

You can customize the appearance of Rectangles by specifying the following optional parameters: ``color``, ``line_width``, ``line_style``, ``fill`` (True or False), and ``fill_alpha``. Here is an example with different styles of Rectangles: ::

    import graphinglib as gl

    rect1 = gl.Rectangle(
        x_bottom_left=2,
        y_bottom_left=2,
        width=10,
        height=10,
        color="red",
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
    figure.add_element(rect1, rect2, rect3)
    figure.display()

There are many useful methods which can be used with Rectangles. For example, you can check if a point is inside a Rectangle: ::

    import graphinglib as gl

    rect = gl.Rectangle(x_bottom_left=0, y_bottom_left=0, width=10, height=10)
    point = gl.Point(5, 5)

    print(point in rect)  # True

You can get the area of a Rectangle and a string representation of its equation: ::

    import graphinglib as gl

    rect = gl.Rectangle(x_bottom_left=0, y_bottom_left=0, width=10, height=10)

    print(rect.area)  # 100
    print(rect.equation)  # 0 <= x <= 10 and 0 <= y <= 10

You can also get Point objects out of a Rectangle, like the center point, or the two points at a given x or y value. You can get these points as tuples or Point objects: ::

    import graphinglib as gl

    rect = gl.Rectangle(x_bottom_left=0, y_bottom_left=0, width=10, height=10)

    point_center = rect.get_center_point(as_point_object=False) # tuple (5, 5)

    point1, point2 = rect.get_points_at_x(5, as_point_object=True) # Point objects

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
        color="red",
        line_width=1,
        line_style="solid",
        fill=True,
        fill_alpha=1,
    )

    circle2 = gl.Circle(
        x_center=4,
        y_center=6,
        radius=7,
        color="blue",
        line_width=2,
        line_style="dashed",
        fill=True,
        fill_alpha=0.5,
    )

    circle3 = gl.Circle(
        x_center=0,
        y_center=-4,
        radius=13,
        color="green",
        line_width=5,
        line_style="dotted",
        fill=False,
    )

    figure = gl.Figure()
    figure.add_element(circle1, circle2, circle3)
    figure.display()

As with Rectangles, there are also many useful methods which can be used with Circles. Here are some examples: ::

    import graphinglib as gl

    circle = gl.Circle(x_center=0, y_center=0, radius=1)

    print(circle.area()) # 3.141592653589793
    print(circle.circumference()) # 6.283185307179586
    print(circle.get_equation())  # (x - 0)^2 + (y - 0)^2 = 1^2

    point = gl.Point(5, 5)
    print(point in circle)  # False

You can also get Point objects out of a Circle like so: ::

    import graphinglib as gl

    circle = gl.Circle(x_center=0, y_center=0, radius=1)

    # Get the center point
    point_center = circle.get_center_point(as_point_object=True) # Point(0, 0)

    # Get the two points at a given x value
    point1, point2 = circle.get_points_at_x(0, as_point_object=False) # tuples (0, 1) and (0, -1)

    # Get the point on the circle at a given angle
    point = circle.get_point_at_angle(45, degrees=True, as_point_object=True) # Point(0.7071067811865476, 0.7071067811865476)

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
    fig.add_element(arrow1, arrow2, arrow3)
    fig.add_element(point1, point2, point3)
    fig.add_element(point4, point5, point6)
    fig.display()

