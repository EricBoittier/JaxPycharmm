#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White}
camera {perspective
  right -3.07*x up 4.61*y
  direction 100.00*z
  location <0,0,100.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7 ambient 0.4 diffuse 0.55}
#declare pale = finish {ambient 0.9 diffuse 0.30 roughness 0.001 specular 0.2 }
#declare intermediate = finish {ambient 0.4 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.2 diffuse 0.80 phong 0.25 phong_size 10.0 specular 0.2 roughness 0.1}
#declare jmol = finish {ambient 0.4 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.2 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.4 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.4 diffuse 0.35 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.3 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.050;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(<  0.14,   0.66,  -0.87>, 0.76, rgb <0.56, 0.56, 0.56>, 0.0, jmol) // #0
atom(< -0.33,  -0.31,  -2.03>, 0.76, rgb <0.56, 0.56, 0.56>, 0.0, jmol) // #1
atom(<  0.45,  -1.31,  -2.53>, 0.66, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #2
atom(< -0.78,  -1.54,  -1.86>, 0.66, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #3
atom(<  0.89,   1.63,  -1.64>, 0.57, rgb <0.56, 0.88, 0.31>, 0.0, jmol) // #4
atom(< -0.89,   1.35,  -0.43>, 0.57, rgb <0.56, 0.88, 0.31>, 0.0, jmol) // #5
atom(<  0.88,  -0.03,   0.00>, 0.57, rgb <0.56, 0.88, 0.31>, 0.0, jmol) // #6
atom(< -1.13,   0.29,  -2.60>, 0.31, rgb <1.00, 1.00, 1.00>, 0.0, jmol) // #7

// no constraints
