/*********************************************************************************
*     File Name           :     tcm.js
*     Created By          :     qing
*     Creation Date       :     [2018-05-11 03:34]
*     Last Modified       :     [2018-05-11 15:59]
*     Description         :      
**********************************************************************************/
                      
$( function() {
function split( val ) {
  return val.split( /,\s*/ );
}
function extractLast( term ) {
  return split( term ).pop();
}

$( "#symptomSearch" )
  // don't navigate away from the field on tab when selecting an item
  .on( "keydown", function( event ) {
	if ( event.keyCode === $.ui.keyCode.TAB &&
		$( this ).autocomplete( "instance" ).menu.active ) {
	  event.preventDefault();
	}
  })
  .autocomplete({
	minLength: 0,
	source: function( request, response ) {
		$.getJSON( "/_autocomplete/symptoms", {
					query: extractLast( request.term )
				  }, response );
	},
	focus: function() {
	  // prevent value inserted on focus
	  return false;
	},
	select: function( event, ui ) {
	  var terms = split( this.value );
	  // remove the current input
	  terms.pop();
	  // add the selected item
	  terms.push( ui.item.value );
	  // add placeholder to get the comma-and-space at the end
	  terms.push( "" );
	  this.value = terms.join( ", " );
	  return false;
	}
  });
  $( "#searchButton" ).click(function() {
      var text = $("#symptomSearch").val();
	  $.ajax({
			type: "POST",
			//the url where you want to sent the userName and password to
            url: "/_predict_diseases",
            data: JSON.stringify({symptoms: split(text)}),
			contentType: "application/json",
            dataType: 'json',
			success: function (response) {
                $('#diseaseCloud').empty()
                d3.wordcloud()
                  .size([800, 600])
                  .selector('#diseaseCloud')
                  .words(response)
                  .font("Open Sans")
                  .onwordclick(function(d, i) {
                        window.location = "https://www.google.com/search?q=" + d.text;
                  })
                  .start();
			}
	   });
	  $.ajax({
			type: "POST",
			//the url where you want to sent the userName and password to
            url: "/_predict_herbs",
            data: JSON.stringify({symptoms: split(text)}),
			contentType: "application/json",
            dataType: 'json',
			success: function (response) {
                $('#herbCloud').empty()
                d3.wordcloud()
                  .size([800, 600])
                  .selector('#herbCloud')
                  .words(response)
                  .font("Open Sans")
                  .onwordclick(function(d, i) {
                        window.location = "https://www.google.com/search?q=" + d.text;
                  })
                  .start();
			}
	 });
  });
});

