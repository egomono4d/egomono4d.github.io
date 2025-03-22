function showSection(id) {
    // Define content to be loaded dynamically for each section
    const contentMap = {
        'video_4d': `
            <div class="container is-max-desktop">
            <div class="columns is-centered has-text-centered">
                <div class="column is-full  panel-style">
                    <h3 class="title is-4">Dynamic Scene Reconstruction</h3>
                    <p>
                        Our method can reconstruct dynamic scenes in a feed-forward, online manner without per-video
                        optimization.
                    </p>
                    <br>
                    <div id="wrapper" style="
              display: flex;
              flex-wrap: nowrap;
              justify-content: center;
              align-items: center;
              gap: 2em;
            ">
                        <div class="iframe-container">
                        <iframe class="vframe"
                            src="https://cut3r.github.io/build/?playbackPath=/recordings/006545_mpii_test.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=0.0,1.0,0.0&baseSpeed=0.8"
                            loading="lazy"></iframe>
                            <button onclick="openInNewTab(this)" data-link="https://cut3r.github.io/build/?playbackPath=/recordings/006545_mpii_test.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=0.0,1.0,0.0&synchronizedVideoOverlay=/recordings/006545_mpii_test.mp4&baseSpeed=0.8">Open in New Tab for Better Visualization</button>
                        </div>
                        <div class="iframe-container">
                        <iframe class="vframe"
                            src="https://cut3r.github.io/build/?playbackPath=/recordings/06eb2803.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=0.0,1.0,0.0&baseSpeed=0.8"
                            loading="lazy"></iframe>
                            <button onclick="openInNewTab(this)" data-link="https://cut3r.github.io/build/?playbackPath=/recordings/06eb2803.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=0.0,1.0,0.0&synchronizedVideoOverlay=/recordings/06eb2803.mp4&baseSpeed=0.8">Open in New Tab for Better Visualization</button>
                        </div>
                    </div>
                    <br>
                    <!-- <div>
            (Results are downsampled for efficient online rendering)
          </div> -->
                    <div class="hide-on-touchscreens" style="
            display: flex;
            justify-content: center;
            gap: 1.5em;
            padding-top: 0.5em;
          ">
                        <div>
                            <i class="ti ti-view-360-arrow"></i> <strong>Left-click</strong> and
                            drag to rotate
                        </div>
                        <div>
                            <i class="ti ti-arrows-move"></i> <strong>Right-click</strong> and
                            drag or
                            <strong>WASD</strong>
                            to move
                        </div>
                        <div><i class="ti ti-zoom"></i> <strong>Scroll</strong> to zoom</div>
                        <div><strong>Click <i class="fas fa-pause"></i></strong> to pause</div>
                    </div>
                    <button onclick="window.location.href='./interactive.html';" target="_blank"
        style="font-size: 20px; font-family: 'Arial', sans-serif; background-color: #92A8D1; color: white; margin: 20px auto; display: block; padding: 15px 15px; border: none; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); text-align: center; transition: all 0.3s ease; cursor: pointer;">
    <span class="icon">
                                            <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 512 512"
                                                style="fill: #ffffff">
                                                <path
                                                    d="M256 0c-25.3 0-47.2 14.7-57.6 36c-7-2.6-14.5-4-22.4-4c-35.3 0-64 28.7-64 64V261.5l-2.7-2.7c-25-25-65.5-25-90.5 0s-25 65.5 0 90.5L106.5 437c48 48 113.1 75 181 75H296h8c1.5 0 3-.1 4.5-.4c91.7-6.2 165-79.4 171.1-171.1c.3-1.5 .4-3 .4-4.5V160c0-35.3-28.7-64-64-64c-5.5 0-10.9 .7-16 2V96c0-35.3-28.7-64-64-64c-7.9 0-15.4 1.4-22.4 4C303.2 14.7 281.3 0 256 0zM240 96.1c0 0 0-.1 0-.1V64c0-8.8 7.2-16 16-16s16 7.2 16 16V95.9c0 0 0 .1 0 .1V232c0 13.3 10.7 24 24 24s24-10.7 24-24V96c0 0 0 0 0-.1c0-8.8 7.2-16 16-16s16 7.2 16 16v55.9c0 0 0 .1 0 .1v80c0 13.3 10.7 24 24 24s24-10.7 24-24V160.1c0 0 0-.1 0-.1c0-8.8 7.2-16 16-16s16 7.2 16 16V332.9c-.1 .6-.1 1.3-.2 1.9c-3.4 69.7-59.3 125.6-129 129c-.6 0-1.3 .1-1.9 .2H296h-8.5c-55.2 0-108.1-21.9-147.1-60.9L52.7 315.3c-6.2-6.2-6.2-16.4 0-22.6s16.4-6.2 22.6 0L119 336.4c6.9 6.9 17.2 8.9 26.2 5.2s14.8-12.5 14.8-22.2V96c0-8.8 7.2-16 16-16c8.8 0 16 7.1 16 15.9V232c0 13.3 10.7 24 24 24s24-10.7 24-24V96.1z">
                                                </path>
                                            </svg>
                                        </span>
        Explore more interactive results for dynamic scene reconstruction!
</button>
                    <script>
                        // Get the paragraph element
                        var para = document.getElementById('click-interactive');

                        // Add event listeners for mouse enter and mouse leave
                        para.addEventListener('mouseenter', function () {
                            para.style.fontSize = '22px'; // Increase font size
                            para.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)'; // Enhance shadow
                        });

                        para.addEventListener('mouseleave', function () {
                            para.style.fontSize = '18px'; // Reset font size
                            para.style.boxShadow = '0 2px 5px rgba(0,0,0,0.1)'; // Reset shadow
                        });
                    </script>
                    <p> See more dynamic reconstruction results on DAVIS dataset in our <a href="./gallery.html">gallery</a>.</p>
                </div>
            </div>
        </div><br>
        `,
        'video_3d': `
            <div class="container is-max-desktop">
            <div class="columns is-centered has-text-centered">
                <div class="column is-full panel-style">
                    <h2 class="title is-4">3D Reconstruction (Video)</h2>

                    <p>Our method can reconstruct the 3D scenes from videos in a feed-forward, online manner without global
                        optimization.</p>
                    <br>
                    <div id="wrapper" style="
                    display: flex;
                    flex-wrap: nowrap;
                    justify-content: center;
                    align-items: center;
                    gap: 2em;
                  ">
                        <div class="iframe-container">
                        <iframe class="vframe"
                            src="https://cut3r.github.io/build/?playbackPath=/static/recordings/3d_1_027.viser&initialCameraPosition=-3.44652427,0.10633461,-0.49854992&initialCameraLookAt=-0.16225029,1.00471075,0.03985959&initialCameraUp=-0.09300407,-0.5189403,0.84973596"
                            loading="lazy"></iframe>
                        <button onclick="openInNewTab(this)" data-link="https://cut3r.github.io/build/?playbackPath=/static/recordings/3d_1_027.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=0.0,1.0,0.0&synchronizedVideoOverlay=/static/recordings/3d_1_027.mp4&baseSpeed=0.5">Open in New Tab for Better Visualization</button>
                        </div>
                        <div class="iframe-container">
                        <iframe class="vframe"
                            src="https://cut3r.github.io/build/?playbackPath=/static/recordings/3d_2_018_gd4_2.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=0.0,1.0,0.0"
                            loading="lazy"></iframe>
                        <button onclick="openInNewTab(this)" data-link="https://cut3r.github.io/build/?playbackPath=/static/recordings/3d_2_018_gd4_2.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=0.0,1.0,0.0&synchronizedVideoOverlay=/static/recordings/3d_2_018.mp4&baseSpeed=0.5">Open in New Tab for Better Visualization</button>
                        </div>
                    </div>
                    <br>
                    <div class="hide-on-touchscreens" style="
                  display: flex;
                  justify-content: center;
                  gap: 1.5em;
                  padding-top: 0.5em;
                ">
                        <div>
                            <i class="ti ti-view-360-arrow"></i> <strong>Left-click</strong> and
                            drag to rotate
                        </div>
                        <div>
                            <i class="ti ti-arrows-move"></i> <strong>Right-click</strong> and
                            drag or
                            <strong>WASD</strong>
                            to move
                        </div>
                        <div><i class="ti ti-zoom"></i> <strong>Scroll</strong> to zoom</div>
                        <div><strong>Click <i class="fas fa-pause"></i></strong> to pause</div>
                    </div>
                    <br>

                    <!--<h3 class="title is-5">More Video Reconstruction Results</h3> -->
                    <!--
                    <br>
                    <div style="text-align: center; padding: 5px; font-family: Arial, sans-serif; background-color: #ffffff; width: 100%; box-sizing: border-box; display: flex;">
                        <div style="color: #333; padding: 5px; background-color: #ffffff; margin-right: 4px; border-radius: 5px; flex: 1;">
                            <span style="font-weight: bold;">Following:</span> <span style="font-weight: normal;">Viewpoint following the camera</span>
                        </div>
                        <div style="color: #333; padding: 5px; background-color: #ffffff; margin-left: 4px; border-radius: 5px; flex: 1;">
                            <span style="font-weight: bold;">Fixed:</span> <span style="font-weight: normal;">Viewpoint fixed in space</span>
                        </div>
                    </div>


                    <div id="carousel-results" class="carousel">
                        <div class="item item-vid5">
                            <video poster="" id="" autoplay playsinline muted loop height="450px" width="auto" loading="lazy">
                            <source data-src="./static/videos/gallery/codec_3d/teaser_batch_1_005.mp4" type="video/mp4">
                            </video>
                        </div>
                        <div class="item item-vid5">
                            <video poster="" id="" autoplay playsinline muted loop height="450px" width="auto" loading="lazy">
                            <source data-src="./static/videos/gallery/codec_3d/teaser_batch_1_028.mp4" type="video/mp4">
                            </video>
                        </div>
                        
                        <div class="item item-vid5">
                            <video poster="" id="" autoplay playsinline muted loop height="450px" width="auto" loading="lazy">
                            <source data-src="./static/videos/gallery/codec_3d/seqs_023.mp4" type="video/mp4">
                            </video>
                        </div>
                    </div>
                    <br>
                    -->
                    <p> See more reconstruction results in our <a href="./gallery_3D.html">gallery</a>.</p>
                </div>
            </div>
        </div>
        `,
        'photo': `
            <div class="container is-max-desktop">
            <div class="columns is-centered has-text-centered">
                <div class="column is-full panel-style">
                    <h2 class="title is-4">3D Reconstruction (Photo Collection)</h2>

                    <p>Our method can reconstruct 3D scenes from sparse photo collections in an online manner, processing them as sequences.</p>
                    <br>

                    <div id="wrapper" style="
      
                            flex-wrap: wrap;
                            justify-content: center;
                            align-items: center;
                            gap: 2em;
                        ">
                        <!-- <div class="siframe-class">
                            <div class="image-grid">
                                <img src="./static/recordings/seq_013_0001.png" alt="Image 1" loading="lazy">
                                <img src="./static/recordings/seq_013_0002.png" alt="Image 2" loading="lazy">
                                <img src="./static/recordings/seq_013_0003.png" alt="Image 3" loading="lazy">
                                <img src="./static/recordings/seq_013_0004.png" alt="Image 4" loading="lazy">
                            </div>
                            <div class="iframe-container">
                            <iframe
                                src="https://cut3r.github.io/build/?playbackPath=/static/recordings/seq_013_1_0.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=-1.0,0.0,0.0"
                                loading="lazy">
                            </iframe>
                            <button onclick="openInNewTab(this)" data-link="https://cut3r.github.io/build/?playbackPath=/static/recordings/seq_013_1_0.viser&initialCameraPosition=0.0,-1.0,1.0&initialCameraLookAt=-1.0,0.0,0.0">Open in New Tab</button>
                            </div>
                        </div> -->
                        <div class="siframe-class">
                            <div class="image-grid">
                                <img src="./static/recordings/seqs_017_0001.png" alt="Image 1" loading="lazy">
                                <img src="./static/recordings/seqs_017_0002.png" alt="Image 2" loading="lazy">
                                <img src="./static/recordings/seqs_017_0003.png" alt="Image 3" loading="lazy">
                                <img src="./static/recordings/seqs_017_0004.png" alt="Image 4" loading="lazy">
                            </div>
                            <div class="iframe-container">
                            <iframe
                                src="https://cut3r.github.io/build/?playbackPath=/static/recordings/seqs_017_1_0.viser&initialCameraPosition=-6.0,2.5,5.0&initialCameraLookAt=5.0,8.0,0.0"
                                loading="lazy">
                            </iframe>
                            <button onclick="openInNewTab(this)" data-link="https://cut3r.github.io/build/?playbackPath=/static/recordings/seqs_017_1_0.viser&initialCameraPosition=-6.0,2.5,5.0&initialCameraLookAt=5.0,8.0,0.0">Open in New Tab</button>
                        </div>
                        </div>
                    </div>

                    <div class="hide-on-touchscreens" style="
          display: flex;
          justify-content: center;
          gap: 1.5em;
          padding-top: 0.5em;
        ">
                        <div>
                            <i class="ti ti-view-360-arrow"></i> <strong>Left-click</strong> and
                            drag to rotate
                        </div>
                        <div>
                            <i class="ti ti-arrows-move"></i> <strong>Right-click</strong> and
                            drag or
                            <strong>WASD</strong>
                            to move
                        </div>
                        <div><i class="ti ti-zoom"></i> <strong>Scroll</strong> to zoom</div>
                        <div><strong>Click <i class="fas fa-pause"></i></strong> to pause</div>
                    </div>

                </div>
            </div>
        </div>
        `,
        'unseen': `
        <div class="container is-max-desktop">
            <div class="columns is-centered has-text-centered">
                <div class="column is-full panel-style">
                    <h2 class="title is-4">Inferring Unseen Structure</h2>
                    <p style="max-width: 90%; margin: 0 auto; text-align: left;">

                        In addition to reconstructing scenes from image observations, 
                        our method can infer unseen structures from virtual viewpoints within the reconstructed scene, predicting colored pointmaps at each virtual view.
                    </p>
                    <br>
                    <video width="90%" controls preload="none" poster="./static/images/unseen_poster.jpg">
                        <source src="./static/videos/unseen.mp4" type="video/mp4">
                    </video>
                </div>
            </div>
        </div><br>
        `
    };

    // Hide all sections and remove their content to free up memory
    const sections = document.getElementsByClassName('dynamic-section');
    for (let i = 0; i < sections.length; i++) {
        sections[i].style.display = 'none';
        // sections[i].innerHTML = '';
    }

    // Load content only for the clicked (or default) section
    const selectedSection = document.getElementById(id);
    // selectedSection.innerHTML = contentMap[id] || `<p>No content for section "${id}".</p>`;
    selectedSection.style.display = 'block';

    // if (id === 'video_3d') {
    //     initCarouselResults();
    // }
}

function initCarouselResults() {
    document.querySelectorAll('#carousel-results video source[data-src]').forEach(srcTag => {
        const realSrc = srcTag.getAttribute('data-src');
        srcTag.setAttribute('src', realSrc);
        srcTag.removeAttribute('data-src');
    });

    bulmaCarousel.attach('#carousel-results', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        initialSlide: 1,
        autoplay: false,
        autoplaySpeed: 3000,
        pagination: false
    });
}

// Show default section on page load
document.addEventListener('DOMContentLoaded', function () {
    showSection('video_4d');
});

function openInNewTab(element) {
    var url = element.getAttribute('data-link');
    window.open(url, '_blank').focus();
}