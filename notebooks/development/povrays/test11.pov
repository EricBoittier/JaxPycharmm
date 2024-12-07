#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -4.39*x up 3.40*y
  direction 50.00*z
  location <0,0,50.00> look_at <0,0,0>}


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
atom(< -1.25,   0.20,  -1.19>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -0.13,  -0.49,  -1.88>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #1
atom(<  0.58,   0.54,  -1.63>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #2
atom(<  1.82,  -0.17,  -1.19>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -1.86,  -0.68,  -2.03>, 0.23, rgb <0.00, 0.91, 0.00>, 0.0, jmol) // #4
atom(< -1.55,   0.06,   0.00>, 0.23, rgb <0.00, 0.91, 0.00>, 0.0, jmol) // #5
atom(< -1.51,   1.39,  -1.74>, 0.23, rgb <0.00, 0.91, 0.00>, 0.0, jmol) // #6
atom(<  0.19,  -1.50,  -2.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
cylinder {< -1.25,   0.20,  -1.19>, < -0.69,  -0.15,  -1.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.13,  -0.49,  -1.88>, < -0.69,  -0.15,  -1.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.25,   0.20,  -1.19>, < -0.34,   0.37,  -1.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.58,   0.54,  -1.63>, < -0.34,   0.37,  -1.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.25,   0.20,  -1.19>, < -1.56,  -0.24,  -1.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.86,  -0.68,  -2.03>, < -1.56,  -0.24,  -1.61>, Rbond texture{pigment {color rgb <0.00, 0.91, 0.00> transmit 0.0} finish{jmol}}}
cylinder {< -1.25,   0.20,  -1.19>, < -1.40,   0.13,  -0.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.55,   0.06,   0.00>, < -1.40,   0.13,  -0.59>, Rbond texture{pigment {color rgb <0.00, 0.91, 0.00> transmit 0.0} finish{jmol}}}
cylinder {< -1.25,   0.20,  -1.19>, < -1.38,   0.80,  -1.46>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.51,   1.39,  -1.74>, < -1.38,   0.80,  -1.46>, Rbond texture{pigment {color rgb <0.00, 0.91, 0.00> transmit 0.0} finish{jmol}}}
cylinder {< -0.13,  -0.49,  -1.88>, <  0.23,   0.03,  -1.75>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.58,   0.54,  -1.63>, <  0.23,   0.03,  -1.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.13,  -0.49,  -1.88>, <  0.85,  -0.33,  -1.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.82,  -0.17,  -1.19>, <  0.85,  -0.33,  -1.53>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.13,  -0.49,  -1.88>, < -0.99,  -0.59,  -1.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.86,  -0.68,  -2.03>, < -0.99,  -0.59,  -1.95>, Rbond texture{pigment {color rgb <0.00, 0.91, 0.00> transmit 0.0} finish{jmol}}}
cylinder {< -0.13,  -0.49,  -1.88>, <  0.03,  -0.99,  -1.98>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.19,  -1.50,  -2.09>, <  0.03,  -0.99,  -1.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.58,   0.54,  -1.63>, <  1.20,   0.19,  -1.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.82,  -0.17,  -1.19>, <  1.20,   0.19,  -1.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
// no constraints
