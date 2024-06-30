:html_theme.sidebar_secondary.remove:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item::
      :child-align: center

      .. div:: sd-fs-1 sd-pb-0
         
         GraphingLib |release| Documentation
        
      .. div:: sd-pt-0 sd-pb-4
            
         A Python library for creating publication-quality figures with ease.
         
      .. button-ref:: examples/index
         :color: primary

         Go to Gallery
    
   .. grid-item::

        .. raw:: html

            <style>
                .grid-container {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    grid-gap: 10px;
                }
                .grid-item {
                    text-align: center;
                }
                .sd-fs-1 {
                    font-size: 24px;
                }
                img {
                    transition: opacity 1s ease-in-out;
                    opacity: 1;
                }
                .fade-out {
                    opacity: 0;
                }
                /* Media query for small screens */
                @media (max-width: 768px) {
                    .grid-container {
                        grid-template-columns: repeat(6, 1fr);
                    }
                }
            </style>

            <div class="grid-container">
                <div class="grid-item">
                    <a id="link1" href="examples/contour_from_data.html"><img id="img1" src="_static/contour_from_data_thumb.png" alt="Image 1"></a>
                </div>
                <div class="grid-item">
                    <a id="link2" href="examples/histogram_pdf.html"><img id="img2" src="_static/histogram_pdf_thumb.png" alt="Image 2"></a>
                </div>
                <div class="grid-item">
                    <a id="link3" href="examples/stream_from_data.html"><img id="img3" src="_static/stream_from_data_thumb.png" alt="Image 3"></a>
                </div>
                <div class="grid-item">
                    <a id="link4" href="examples/curve_fill_between.html"><img id="img4" src="_static/curve_fill_between_thumb.png" alt="Image 4"></a>
                </div>
                <div class="grid-item">
                    <a id="link5" href="examples/heatmap_from_data.html"><img id="img5" src="_static/heatmap_from_data_thumb.png" alt="Image 5"></a>
                </div>
                <div class="grid-item">
                    <a id="link6" href="examples/vectorfield_from_function.html"><img id="img6" src="_static/vectorfield_from_function_thumb.png" alt="Image 6"></a>
                </div>
            </div>

            <script>
                const imageSources = [
                    { src: '_static/contour_from_data_thumb.png', link: 'examples/contour_from_data.html' },
                    { src: '_static/contour_from_function_thumb.png', link: 'examples/contour_from_function.html' },
                    { src: '_static/curve_arithmetic_thumb.png', link: 'examples/curve_arithmetic.html' },
                    { src: '_static/curve_fill_between_thumb.png', link: 'examples/curve_fill_between.html' },
                    { src: '_static/curve_from_data_thumb.png', link: 'examples/curve_from_data.html' },
                    { src: '_static/curve_from_function_thumb.png', link: 'examples/curve_from_function.html' },
                    { src: '_static/curve_intersections_thumb.png', link: 'examples/curve_intersections.html' },
                    { src: '_static/curve_slices_thumb.png', link: 'examples/curve_slices.html' },
                    { src: '_static/fit_from_polynomial_thumb.png', link: 'examples/fit_from_polynomial.html' },
                    { src: '_static/heatmap_from_data_thumb.png', link: 'examples/heatmap_from_data.html' },
                    { src: '_static/heatmap_from_function_thumb.png', link: 'examples/heatmap_from_function.html' },
                    { src: '_static/heatmap_from_image_thumb.png', link: 'examples/heatmap_from_image.html' },
                    { src: '_static/histogram_pdf_thumb.png', link: 'examples/histogram_pdf.html' },
                    { src: '_static/histogram_thumb.png', link: 'examples/histogram.html' },
                    { src: '_static/hlines_vlines_thumb.png', link: 'examples/hlines_vlines.html' },
                    { src: '_static/scatter_errorbars_thumb.png', link: 'examples/scatter_errorbars.html' },
                    { src: '_static/stream_from_data_thumb.png', link: 'examples/stream_from_data.html' },
                    { src: '_static/vectorfield_from_data_thumb.png', link: 'examples/vectorfield_from_data.html' },
                    { src: '_static/vectorfield_from_function_thumb.png', link: 'examples/vectorfield_from_function.html' },
                    { src: '_static/curve_error_curves_thumb.png', link: 'examples/curve_error_curves.html'},
                    { src: '_static/dark_style_thumb.png', link: 'examples/dark_style.html' },
                    { src: '_static/fourier_series_thumb.png', link: 'examples/fourier_series.html' },
                    { src: '_static/infinite_square_well_thumb.png', link: 'examples/infinite_square_well.html' },
                    { src: '_static/lorenz_attractor_thumb.png', link: 'examples/lorenz_attractor.html' },
                    { src: '_static/mandelbrot_set_thumb.png', link: 'examples/mandelbrot_set.html' },
                    { src: '_static/shape_unions_thumb.png', link: 'examples/shape_unions.html' },
                ];

                imageSources.forEach((image) => {
                   const img = new Image();
                   img.src = image.src;
                });

                let imageElements = [
                    { img: document.getElementById('img1'), link: document.getElementById('link1') },
                    { img: document.getElementById('img2'), link: document.getElementById('link2') },
                    { img: document.getElementById('img3'), link: document.getElementById('link3') },
                    { img: document.getElementById('img4'), link: document.getElementById('link4') },
                    { img: document.getElementById('img5'), link: document.getElementById('link5') },
                    { img: document.getElementById('img6'), link: document.getElementById('link6') }
                ];

                let lastChangedIndices = [];

                function rotateImages() {
                    // Get the currently displayed images
                    const displayedImages = imageElements.map(el => el.img.src.split('/').pop());

                    // Get the pool of remaining images
                    const remainingImages = imageSources.filter(img => !displayedImages.includes(img.src.split('/').pop()));

                    let newImage, randomImageIndex;
                    let attempts = 0;
                    const maxAttempts = 10;

                    do {
                        // Select a new random image from the remaining pool
                        newImage = remainingImages[Math.floor(Math.random() * remainingImages.length)];

                        // Find a random image element to replace that hasn't been changed recently
                        do {
                            randomImageIndex = Math.floor(Math.random() * imageElements.length);
                        } while (lastChangedIndices.includes(randomImageIndex));

                        attempts++;
                        // Break the loop if max attempts reached, avoid infinite loop
                        if (attempts >= maxAttempts) break;
                    } while (imageElements[randomImageIndex].img.src.split('/').pop() === newImage.src.split('/').pop());

                    const currentElement = imageElements[randomImageIndex];

                    // Add fade-out class to the image element
                    currentElement.img.classList.add('fade-out');

                    // Wait for the fade-out transition to complete
                    setTimeout(() => {
                        // Update the image source and link
                        currentElement.img.src = newImage.src;
                        currentElement.link.href = newImage.link;

                        // Remove fade-out class after updating the image
                        currentElement.img.classList.remove('fade-out');
                    }, 1000); // Duration of the fade-out effect

                    // Update the list of last changed indices
                    lastChangedIndices.push(randomImageIndex);
                    if (lastChangedIndices.length > 3) {
                        lastChangedIndices.shift(); // Keep only the last 3 changed indices
                    }
                }

                setInterval(rotateImages, 5000); // Rotate every 5 seconds
            </script>
      

.. grid:: 1 1 3 3

    .. grid-item-card::
        :img-top: _static/icons//index_getting_started.svg

        Getting started
        ^^^^^^^^^^^^^^^
        If you are new to GraphingLib, check out this section first to learn how to install and import GraphingLib.
        ++++
        
        .. button-ref:: installation
            :expand:
            :color: primary
            :click-parent:

            Get started
    
    .. grid-item-card::
        :img-top: _static/icons/index_user_guide.svg

        Handbook
        ^^^^^^^^
        Once GraphingLib is installed, visit this section to learn how to use its different features.
        ++++

        .. button-ref:: handbook/index
            :expand:
            :color: primary
            :click-parent:

            Visit the Handbook
        
    .. grid-item-card::
        :img-top: _static/icons/index_api.svg
        
        Reference
        ^^^^^^^^^
        If you are looking for more details on objects and their methods, visit this section.
        ++++

        .. button-ref:: api
            :expand:
            :color: primary
            :click-parent:

            Visit the API Reference

GraphingLib: Making Data Visualization Easy
-------------------------------------------

GraphingLib is an open-source data visualization library in Python, designed to make plotting and data analysis both intuitive and efficient. Built on the robust foundations of Matplotlib, GraphingLib enhances readability, conciseness, and user-friendliness in the creation of figures.

Why Choose GraphingLib?
~~~~~~~~~~~~~~~~~~~~~~~

**Object-Oriented Design**

GraphingLib simplifies Matplotlib’s complex API by introducing an object-oriented approach. Each element on the graph is an object with properties you can set and modify at any time, ensuring clean, intuitive code.

**Integrated Data Analysis**

GraphingLib goes beyond basic plotting. Perform curve fits, differentiation, integration, intersections, and other standard operations directly on Curve and Scatter objects. Calculate statistical properties of histograms. Use set operations on polygons. All these features are built into the library using the powerful capabilities of NumPy, SciPy, and Shapely.

**User Defined Figure Styles**

GraphingLib allows you to apply prepackaged or custom styles effortlessly. Our :doc:`GUI Style Editor <handbook/figure_style_file>` lets you create, modify, and save styles, which can then be applied with a simple keyword. You can even set your own style as the default for all your figures. With GraphingLib, you can customize your plots quickly while retaining full control over their appearance.

.. toctree::
   :maxdepth: 3
   :hidden:
   
   installation
   handbook/index
   Gallery <examples/index>
   API <api>
   Contributing <contributing>
   Releases <release_notes/index>
   Style Editor <https://www.graphinglib.org/projects/graphinglibstyleeditor/>
