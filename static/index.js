(function () {
    var canvas = document.querySelector("#canvas");
    var context = canvas.getContext("2d");
    canvas.width = 280;
    canvas.height = 280;

    var Mouse = {x: 0, y: 0};
    var lastMouse = {x: 0, y: 0};
    var Touch = {x: 0, y: 0};
    var lastTouch = {x: 0, y: 0};
    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.color = "white";
    context.lineWidth = 10;
    context.lineJoin = context.lineCap = 'round';

    debug();
    // 移动端
    canvas.addEventListener("ontouchmove", function (e) {
        lastTouch.x = Touch.x;
        lastTouch.y = Touch.y;

        Touch.x = e.clientX - this.offsetLeft - 15;
        Touch.y = e.clientY - this.offsetTop - 15;
    }, false);

    canvas.addEventListener("ontouchstart", function (e) {
        canvas.addEventListener("ontouchmove", onPaint, false);
    }, false);

    canvas.addEventListener("ontouchend", function (e) {
        canvas.removeEventListener("ontouchmove", onPaint, false);
    }, false);

    var onPaint = function () {
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo(lastTouch.x, lastTouch.y);
        context.lineTo(Touch.x, Touch.y);
        context.closePath();
        context.stroke();
    };



        // 鼠标事件
    canvas.addEventListener("mousemove", function (e) {
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;

        Mouse.x = e.pageX - this.offsetLeft - 15;
        Mouse.y = e.pageY - this.offsetTop - 15;
    }, false);

    canvas.addEventListener("mousedown", function (e) {
        canvas.addEventListener("mousemove", onPaint, false);
    }, false);

    canvas.addEventListener("mouseup", function () {
        canvas.removeEventListener("mousemove", onPaint, false);
    }, false);

    var onPaint = function () {
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
        context.closePath();
        context.stroke();


    };






    function debug() {
        $("#clearButton").on("click", function () {
            context.clearRect(0, 0, 280, 280);
            context.fillStyle = "black";
            context.fillRect(0, 0, canvas.width, canvas.height);
        });
    }
}());
