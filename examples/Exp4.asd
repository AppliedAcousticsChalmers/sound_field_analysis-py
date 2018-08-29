<?xml version="1.0" encoding="utf-8"?>
<asdf>
  <header>
    <name>Example for rendering of spherical microphone array data</name>
    <description>
      Note that the sampling frequency is 48 kHz.
      
      Command for invoking SSR from command line (when SSR is installed): 
      	ssr-brs Exp4.asd
      	
      When SSR is not installed but the app bundle is used on macOS:
		open -a SoundScapeRenderer --args --brs "/Users/jens.ahrens/Documents/coding/sound_field_analysis-py/examples/Exp4.asd"      
      	
    </description>
  </header>

  <scene_setup>
    <volume>0</volume>

    <source name="Static source" properties_file="SSR_IRs.wav" volume="-12">
      <file channel="1">/Users/jens.ahrens/Documents/audio/ssr_scenes/audio/Track49_48k.wav</file>
      <position x="0" y="2" fixed="true"/>
    </source>

  </scene_setup>
</asdf>
