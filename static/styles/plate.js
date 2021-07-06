function refresh(node)
{
   var times = 500;

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
  getPlate();
}

function getPlate() {
    var
        $http,
        $self = arguments.callee;

    if (window.XMLHttpRequest) {
        $http = new XMLHttpRequest();
    } else if (window.ActiveXObject) {
        try {
            $http = new ActiveXObject('Msxml2.XMLHTTP');
        } catch(e) {
            $http = new ActiveXObject('Microsoft.XMLHTTP');
        }
    }

    if ($http) {
        $http.onreadystatechange = function()
        {
            if (/4|^complete$/.test($http.readyState)) {
                document.getElementById('plate_number').innerHTML = $http.responseText;
                setTimeout(function(){$self();}, 500);
            }
        };
        $http.open('GET', './static/texts/plate_number_0.txt' + '?' + new Date().getTime(), true);
        $http.send(null);
    }

}

// function run() {
//     fetch("http://127.0.0.1:5000/plate", {
//     method: 'GET'
//     })
// }