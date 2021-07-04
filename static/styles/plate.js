function refresh(node)
{
   var times = 300; // gap in Milli Seconds;

   (function startRefresh()
   {
      var address;
      if(node.src.indexOf('?')>-1)
       address = node.src.split('?')[0];
      else 
       address = node.src;
      node.src = address+"?time="+new Date().getTime();

      setTimeout(startRefresh,times);
   })();

}

window.onload = function()
{
  var node = document.getElementById('img');
  refresh(node);
}


function getPlate() {
    $.ajax({
        url: 'static/texts/plate_number_0.txt',
        dataType: 'text',
        success: function(text) {
            $("#plate_number").text(text);
            document.getElementById('plate_number').innerText = text
            setTimeout(getPlate, 300);
        }
    })
}

// function run() {
//     fetch("http://127.0.0.1:5000/plate", {
//     method: 'GET'
//     })
// }