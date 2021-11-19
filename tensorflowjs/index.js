
var video;
var videoCanvas;

var model;
loadModel = async function () {
    model = await tf.loadGraphModel('model/model.json');

    var camera = document.getElementById('camera_id');
    camera.addEventListener('click', onClickCamera);
}

var onCameraFrame = async function () {

    const context = videoCanvas.getContext('2d');
    context.drawImage(video, 0, 0);
    var example = tf.browser.fromPixels(videoCanvas);
    example2 = tf.image.resizeBilinear(example, [224, 224]);
    var example3 = example2.expandDims(0);
    const output = await model.predict(example3);
    example2.dispose();
    example.dispose();
    example3.dispose();
    console.log(output);
    output.dispose();
    window.requestAnimationFrame(onCameraFrame);
}


var onClickCamera = function () {

    if (!video) {
        video = document.getElementById('video_id');

        videoCanvas = document.createElement('canvas');
        videoCanvas.width = '400';
        videoCanvas.height = '400';
        document.body.appendChild(videoCanvas);
    }

    var facingMode = "environment"; // Can be 'user' or 'environment' to access back or front camera (NEAT!)
    var constraints = {
        audio: false,
        video: {
            facingMode: facingMode
        },
        autoplay: true,
        play: true
    };

    navigator.mediaDevices.
        getUserMedia(constraints).
        then(function success(stream) {
            video.srcObject = stream;
            video.play();
            setTimeout(() => {
                window.requestAnimationFrame(onCameraFrame);
            }, 1000);
        });
}


loadModel();