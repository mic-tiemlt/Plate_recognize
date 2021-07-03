function update(img_path) {
    var source = img_path,
        timestamp = (new Date()).getTime(),
        newUrl = source + '?_=' + timestamp;
    document.getElementById("img").src = newUrl;
    document.getElementById("img1").src =  newUrl;
    setTimeout(update, 1000);
}