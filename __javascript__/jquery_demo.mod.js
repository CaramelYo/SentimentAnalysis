	(function () {
		var random = {};
		__nest__ (random, '', __init__ (__world__.random));
		var start = function () {
			var changeColors = function () {
				var __iterable0__ = $divs;
				for (var __index0__ = 0; __index0__ < __iterable0__.length; __index0__++) {
					var div = __iterable0__ [__index0__];
					$ (div).css (dict ({'color': 'rgb({},{},{})'.format.apply (null, function () {
						var __accu0__ = [];
						for (var i = 0; i < 3; i++) {
							__accu0__.append (int (256 * Math.random ()));
						}
						return __accu0__;
					} ())}));
				}
			};
			var $divs = $ ('div');
			changeColors ();
			window.setInterval (changeColors, 500);
		};
		__pragma__ ('<use>' +
			'random' +
		'</use>')
		__pragma__ ('<all>')
			__all__.start = start;
		__pragma__ ('</all>')
	}) ();
