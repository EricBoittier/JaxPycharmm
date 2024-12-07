#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -3.51*x up 3.83*y
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
atom(< -0.32,   0.35,  -1.21>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(<  0.56,  -0.33,  -2.32>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #1
atom(<  0.29,  -1.56,  -2.67>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #2
atom(< -0.02,  -0.64,  -3.55>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(<  0.29,   0.40,   0.00>, 0.23, rgb <0.00, 0.91, 0.00>, 0.0, jmol) // #4
atom(< -0.59,   1.60,  -1.49>, 0.23, rgb <0.00, 0.91, 0.00>, 0.0, jmol) // #5
atom(< -1.44,  -0.36,  -1.05>, 0.23, rgb <0.00, 0.91, 0.00>, 0.0, jmol) // #6
atom(<  1.55,   0.14,  -2.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
cylinder {< -0.32,   0.35,  -1.21>, <  0.12,   0.01,  -1.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.56,  -0.33,  -2.32>, <  0.12,   0.01,  -1.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.32,   0.35,  -1.21>, < -0.01,   0.37,  -0.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.29,   0.40,   0.00>, < -0.01,   0.37,  -0.60>, Rbond texture{pigment {color rgb <0.00, 0.91, 0.00> transmit 0.0} finish{jmol}}}
cylinder {< -0.32,   0.35,  -1.21>, < -0.45,   0.97,  -1.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.59,   1.60,  -1.49>, < -0.45,   0.97,  -1.35>, Rbond texture{pigment {color rgb <0.00, 0.91, 0.00> transmit 0.0} finish{jmol}}}
cylinder {< -0.32,   0.35,  -1.21>, < -0.88,  -0.01,  -1.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.44,  -0.36,  -1.05>, < -0.88,  -0.01,  -1.13>, Rbond texture{pigment {color rgb <0.00, 0.91, 0.00> transmit 0.0} finish{jmol}}}
cylinder {<  0.56,  -0.33,  -2.32>, <  0.43,  -0.95,  -2.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.29,  -1.56,  -2.67>, <  0.43,  -0.95,  -2.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.56,  -0.33,  -2.32>, <  0.27,  -0.48,  -2.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.02,  -0.64,  -3.55>, <  0.27,  -0.48,  -2.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.56,  -0.33,  -2.32>, <  1.06,  -0.09,  -2.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.55,   0.14,  -2.52>, <  1.06,  -0.09,  -2.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.29,  -1.56,  -2.67>, <  0.14,  -1.10,  -3.11>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.02,  -0.64,  -3.55>, <  0.14,  -1.10,  -3.11>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
// no constraints
