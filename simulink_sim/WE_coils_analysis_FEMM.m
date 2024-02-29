%% verify femm path
if not(exist('openfemm','file'))
    if exist('C:\\Program Files (x86)\\femm42\\mfiles','dir')
        addpath('C:\\Program Files (x86)\\femm42\\mfiles');
    else
        warndlg('Missing FEMM4.2 or path to (\mfile) directory');
        return
    end
end
clear variables
clc

%% Parametri della ferrite
mu_Ferrite=600;
sigma_ferrite=0;
% frequenza
f = 85e3;

%% numero spire coil
Nt=15;
Nr=20;
%% FEMM open setup
openfemm    
%open new magnetostatic problem
newdocument(0) %0=magnetostatic problem
%load materials from materials library
mi_getmaterial('Air')
mi_getmaterial('Copper')
%user defined materials
mi_addmaterial('Ferrite',mu_Ferrite,mu_Ferrite,0,0,sigma_ferrite,0,0,0,0,0,0,0,0);
%define problem
mi_probdef(f,'meters','axi',1E-8,0,30,0)

%circuit definition
Itr = 1;
Ire = 0;
mi_addcircprop('Tr',Itr,1) 
mi_addcircprop('Re',Ire,1)

%geometria transmitter
t_fe_rad = 37.5e-3; %raggio ferrite 
t_fe_h = 5e-4; %altezza ferrite
t_coil_rad = 0.034; %raggio coil
t_coil_h = 1e-3; %altezza coil
t_coilfe_dist = 0; %distanza coil ferrite
t_int_rad = 0.017;
t_ext_rad = 0.034;

%geometria receiver
r_fe_rad = 0.025; %raggio ferrite 
r_fe_h = 5e-4; %altezza ferrite
r_coil_rad = 0.0225; %raggio coil
r_coil_h = 2e-3; %altezza coil
r_coilfe_dist = 0; %distanza coil ferrite
r_int_rad = 0.01;
r_ext_rad = 0.0225;

%gap
gap = 5e-3; %distanza coil

%% drawing receiver
%piatto ferrite
zr1 = gap+r_coil_h+r_coilfe_dist;
rr1 = 0;
zr2 = zr1;
rr2 = r_fe_rad;
zr3 = gap+r_coil_h+r_coilfe_dist+r_fe_h;
rr3 = rr2;
zr4 = zr3;
rr4 = 0;
mi_addnode(rr1,zr1)
mi_addnode(rr2,zr2)
mi_addnode(rr3,zr3)
mi_addnode(rr4,zr4)

mi_addsegment(rr1,zr1,rr2,zr2)
mi_addsegment(rr2,zr2,rr3,zr3)
mi_addsegment(rr3,zr3,rr4,zr4)
mi_addsegment(rr4,zr4,rr1,zr1)

% assign material
rcenter = rr2*0.5;
zcenter = (zr4+zr1)*0.5;
mi_addblocklabel(rcenter,zcenter); %punto interno al rettangolo
mi_selectlabel(rcenter,zcenter);
mi_setblockprop('Ferrite',1,0,'',30,0,0) %automesh
mi_clearselected();

%coil
zcr1 = gap;
rcr1 = r_int_rad;
zcr2 = zcr1;
rcr2 = r_ext_rad;
zcr3 = zcr2+r_coil_h;
rcr3 = rcr2;
zcr4 = zcr3;
rcr4 = rcr1;
mi_addnode(rcr1,zcr1)
mi_addnode(rcr2,zcr2)
mi_addnode(rcr3,zcr3)
mi_addnode(rcr4,zcr4)

mi_addsegment(rcr1,zcr1,rcr2,zcr2)
mi_addsegment(rcr2,zcr2,rcr3,zcr3)
mi_addsegment(rcr3,zcr3,rcr4,zcr4)
mi_addsegment(rcr4,zcr4,rcr1,zcr1)

% assign material
rcenter = (rcr1+rcr2)*0.5;
zcenter = (zcr4+zcr1)*0.5;
r_eval = rcenter;
z_eval = zcenter;
mi_addblocklabel(rcenter,zcenter); %punto interno alla circonferenza
mi_selectlabel(rcenter,zcenter);
mi_setblockprop('Copper',1,0,'Re',30,0,Nr);
mi_clearselected();

%% drawing transmitter
%piatto ferrite
zt1 = -(t_coil_h+t_coilfe_dist);
rt1 = 0;
zt2 = zt1;
rt2 = t_fe_rad;
zt3 = -(t_coil_h+t_coilfe_dist+t_fe_h);
rt3 = rt2;
zt4 = zt3;
rt4 = 0;
mi_addnode(rt1,zt1)
mi_addnode(rt2,zt2)
mi_addnode(rt3,zt3)
mi_addnode(rt4,zt4)

mi_addsegment(rt1,zt1,rt2,zt2)
mi_addsegment(rt2,zt2,rt3,zt3)
mi_addsegment(rt3,zt3,rt4,zt4)
mi_addsegment(rt4,zt4,rt1,zt1)

% assign material
rcenter = rt2*0.5;
zcenter = (zt4+zt1)*0.5;
mi_addblocklabel(rcenter,zcenter); %punto interno alla circonferenza
mi_selectlabel(rcenter,zcenter);
mi_setblockprop('Ferrite',1,0,'',30,0,0) %automesh
mi_clearselected();

%coil
zct1 = 0;
rct1 = t_int_rad;
zct2 = zct1;
rct2 = t_ext_rad;
zct3 = -(zct2+t_coil_h);
rct3 = rct2;
zct4 = zct3;
rct4 = rct1;
mi_addnode(rct1,zct1)
mi_addnode(rct2,zct2)
mi_addnode(rct3,zct3)
mi_addnode(rct4,zct4)

mi_addsegment(rct1,zct1,rct2,zct2)
mi_addsegment(rct2,zct2,rct3,zct3)
mi_addsegment(rct3,zct3,rct4,zct4)
mi_addsegment(rct4,zct4,rct1,zct1)

% assign material
rcenter = (rct1+rct2)*0.5;
zcenter = (zct4+zct1)*0.5;
mi_addblocklabel(rcenter,zcenter); %punto interno alla circonferenza
mi_selectlabel(rcenter,zcenter);
mi_setblockprop('Copper',1,0,'Tr',30,0,Nt);
mi_clearselected();

%draw domain
mi_addnode(0,r_fe_rad*5);
mi_addnode(0,-r_fe_rad*5);
mi_addarc(0,-r_fe_rad*5,0,r_fe_rad*5,180,1);
mi_addsegment(0,-r_fe_rad*5,0,r_fe_rad*5);

% assign material
mi_addblocklabel(r_coil_rad,r_fe_rad*4); %punto interno alla circonferenza
mi_selectlabel(r_coil_rad,r_fe_rad*4);
mi_setblockprop('Air',1,0,'',30,0,1);
mi_clearselected();

%save file
mi_saveas('coil_test.fem');
%run sim
mi_analyse(0)
%open post-processor
mi_loadsolution();
%seleziona baricentro coil re
mo_selectblock(r_eval,z_eval) 
M = Nr*abs(mo_blockintegral(1)/mo_blockintegral(5))/Itr %flux times primary current (mo_blockintegral(1) evaluates the magnetic vector potential multipled by the cross-sectional area; mo_blockintegral(5) evaluates the cross-sectional area)
mo_clearblock()
%seleziona baricentro coil tr
mo_selectblock(rcenter,zcenter) 
L1 = abs(mo_blockintegral(0)/Itr^2) %flux times primary current (mo_blockintegral(1) evaluates the magnetic vector potential multipled by the cross-sectional area; mo_blockintegral(5) evaluates the cross-sectional area)
mo_clearblock()


