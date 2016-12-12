
console.log("YO");
audioFile = "chimes.wav";
shell = new ActiveXObject("Wscript.Shell");
//command = "sndrec32 /play /close \"" + audioFile + "\"";
command = "soundRecorder /play /close \"" + audioFile + "\"";
command = "start " + audioFile;
WScript.echo(command);
shell.Run(command, 0);