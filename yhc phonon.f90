program main
USE MSIMSL
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
real:: real_site(2,50),relative_site(2,50),kk_site(2,150),Rij(50,50),KK(3,3),theta
real:: U1(3,3),U2(3,3),kkk(3,3),kkkk(3,3),bzzR(150,6),Ra,Ib,dx,dy
integer::jl,i,j,ii,jj
complex::bzz(6),D(6,6),kt(4,3,3),Cab
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!..............................产生第一布里渊区k值..............................
open(1,FILE='k_site.dat')
do i=1,50
kk_site(1,i)=i/100.0+0
kk_site(2,i)=0
enddo
do i=1,50
kk_site(1,i+50)=kk_site(1,50)+(0.33333-0.5)*i/50
kk_site(2,i+50)=0.6667*i/50
enddo
do i=1,50
kk_site(1,i+100)=kk_site(1,100)-0.3333*i/50
kk_site(2,i+100)=kk_site(2,100)-0.6667*i/50
enddo
write(1,"(2f6.3)") kk_site
close(1)
!..............................产生实空间的坐标..............................!
open(2,FILE='real_site.dat')
k=0
do i = -2,2
   do j = -2,2
   k=k+1
   if(i==0.and.j==0)print*,k,k+25
   relative_site(1,k) = 1.0/3+i
   relative_site(2,k) = 2.0/3+j
   relative_site(1,k+25) = 2.0/3+i
   relative_site(2,k+25) = 1.0/3+j

   real_site(1,k) = relative_site(1,k)*2.46+relative_site(2,k)*2.46*cosd(120.)
   real_site(2,k) = relative_site(2,k)*cosd(30.)*2.46
   real_site(1,k+25) = relative_site(1,k+25)*2.46+relative_site(2,k+25)*2.46*cosd(120.)
   real_site(2,k+25) = relative_site(2,k+25)*cosd(30.)*2.46

   enddo
enddo
write(2,"(f6.3,2x,f6.3)")  real_site
close(2)
!..............................筛选距离..............................!
open(11,FILE='DD.dat')
open(22,FILE='bzz.dat')

do k=1,150      !考察每个K点。。。。。。。。。。。。。。。。。。。。。。。
kt=0
kk=0
 do i=13,38,25  !统计胞内第i个原子（A,B）
do j=1,50       !统计所有原子对第i个的相互作用..............
  dx=real_site(1,i)-real_site(1,j)
  dy=real_site(2,i)-real_site(2,j)
  Rij(i,j)=dx**2+dy**2
  Rij(i,j)=sqrt(Rij(i,j))

if(abs(Rij(i,j)-1.42)<0.01)  jl=1  !-----------------------------!
if(abs(Rij(i,j)-2.46)<0.01)  jl=2  !                             !
if(abs(Rij(i,j)-2.84)<0.01)  jl=3  !根据距离判断是第几近邻并标记 !
if(abs(Rij(i,j)-3.758)<0.01) jl=4  !                             !
if(abs(Rij(i,j)-0)<0.01)     jl=0  !-----------------------------!
if(Rij(i,j)>4) jl=5
!--------------------------根据i-j近邻类型，判断要不要统计，并求出相应力常数矩阵KK
select case(jl)
case(1)
kk(1,1)=36.5 ;kk(2,2)=24.50;kk(3,3)=9.82
case(2)
kk(1,1)=8.80 ;kk(2,2)=-3.23;kk(3,3)=-0.4
case(3)
kk(1,1)=3.00 ;kk(2,2)=-5.25;kk(3,3)=0.15
case(4)
kk(1,1)=-1.92;kk(2,2)=2.29 ;kk(3,3)=-0.58
case default
kk=0
!cycle  !不统计--这一类J原子--------------------
endselect
!----------------------求紧邻向量与X轴夹角
  dx=real_site(1,i)-real_site(1,j)
  dy=real_site(2,i)-real_site(2,j)
  if(dx==0)then
    if(dy>0.0)theta=acos(-1.0)/2
  if(dy<0.0)theta=-acos(-1.0)/2
else
theta=abs(dy/dx)                    !原来角度求解有问题!!!!
theta=atan(theta)
if(dx>0.0.and.dy>0.0)theta=theta
if(dx<0.0.and.dy>0.0)theta=acos(-1.0)-theta
if(dx<0.0.and.dy<0.0)theta=-acos(-1.0)+theta
if(dx>0.0.and.dy<0.0)theta=-theta
endif

!..............................U1为原矩阵，U2为逆矩阵..............................!
U1(1,1)=cos(theta);u1(1,2)=sin(theta);u1(1,3)=0
U1(2,1)=-sin(theta);u1(2,2)=cos(theta);u1(2,3)=0
U1(3,1)=0         ;u1(3,2)=0         ;u1(3,3)=1

U2(1,1)=cos(theta) ;u2(1,2)=-sin(theta);u2(1,3)=0
U2(2,1)=sin(theta);u2(2,2)=cos(theta) ;u2(2,3)=0
U2(3,1)=0          ;u2(3,2)=0          ;u2(3,3)=1
!-------------------------------两步法旋转KK 1:U2*KK------------------
do ii=1,3
  do jj=1,3
  kkk(ii,jj)=U2(ii,1)*kk(1,jj)+U2(ii,2)*kk(2,jj)+U2(ii,3)*kk(3,jj)
  !if(i==13)   kkk(ii,jj)=U2(ii,1)*kk(1,jj)+U2(ii,2)*kk(2,jj)+U2(ii,3)*kk(3,jj)
  !if(i==38)   kkk(ii,jj)=U1(ii,1)*kk(1,jj)+U1(ii,2)*kk(2,jj)+U1(ii,3)*kk(3,jj)
  enddo
 enddo 
!-------------------------------两步法旋转KK 1:KK*U1------------------  
do ii=1,3
  do jj=1,3
 kkkk(ii,jj)=kkk(ii,1)*u1(1,jj)+kkk(ii,2)*u1(2,jj)+kkk(ii,3)*u1(3,jj) 
 !if(i==13)kkkk(ii,jj)=kkk(ii,1)*u1(1,jj)+kkk(ii,2)*u1(2,jj)+kkk(ii,3)*u1(3,jj)
 !if(i==38)kkkk(ii,jj)=kkk(ii,1)*u2(1,jj)+kkk(ii,2)*u2(2,jj)+kkk(ii,3)*u2(3,jj)
  enddo
 enddo 
!====================================================================
 if(0)then
 print*,'---------------------',i,j,theta*180/acos(-1.0),"---------------"
 write(*,'(f6.3,2x,f6.3,2x,f6.3,2x)')kk
 print*,'---------------------'
 write(*,'(f6.3,2x,f6.3,2x,f6.3,2x)')kkk
 print*,'---------------------'
 write(*,'(f6.3,2x,f6.3,2x,f6.3,2x)')kkkk
 print*,'---------------------'
 write(*,'(f6.3,2x,f6.3,2x,f6.3,2x)')u1
 print*,'---------------------'
 write(*,'(f6.3,2x,f6.3,2x,f6.3,2x)')u2
pause
endif

!----------------------i=13,38;;;j=1,50------------
if(i==13)ij=1
if(i==38)ij=2
do ii=1,3
  do jj=1,3
!if(j<=25) then
  !!print*,ii,jj,ij,k,j
  !!--------delta(A,B)=0;delta(A,A)=1;delta(B,B)=1;delta(B,A)=0
  !!Pi=acos(-1.0)而不是asin(1.0)
  if(i==13)then  !!AA----------------
     if(j>25)tt=0
     if(j<=25)tt=1
     Ra=kkkk(ii,jj)-tt*kkkk(ii,jj)*cos(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Ib=-tt*kkkk(ii,jj)*sin(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Cab=cmplx(Ra,Ib)
   Kt(1,II,jj)=Kt(1,II,jj)+Cab 
  endif !!AA----------------

if(i==13)then  !!AB----------------
     if(j<=25)tt=0
     if(j>25)tt=1
     Ra=-tt*kkkk(ii,jj)*cos(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Ib=-tt*kkkk(ii,jj)*sin(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Cab=cmplx(Ra,Ib)
   Kt(2,II,jj)=Kt(2,II,jj)+Cab 
  endif !!AB---------------- 

if(i==38)then  !!BA----------------
     if(j<=25)tt=1
     if(j>25)tt=0
     Ra=-tt*kkkk(ii,jj)*cos(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Ib=-tt*kkkk(ii,jj)*sin(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Cab=cmplx(Ra,Ib)
   Kt(3,II,jj)=Kt(3,II,jj)+Cab 
  endif !!AB----------------

  if(i==38)then  !!BB----------------
     if(j<=25)tt=0
     if(j>25)tt=1
     Ra=kkkk(ii,jj)-tt*kkkk(ii,jj)*cos(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Ib=-tt*kkkk(ii,jj)*sin(dx*kk_site(1,k)*2*acos(-1.0)+dy*kk_site(2,k)*2*acos(-1.0))
     Cab=cmplx(Ra,Ib)
   Kt(4,II,jj)=Kt(4,II,jj)+Cab 
  endif !!BB----------------
  enddo
  enddo
!------------------------
  enddo  !..............................do j=1,M 遍历其他相互作用原子..............................
enddo  !..............................统计的do i=13,38: A,B   完成小对角矩阵KA(i,j) and KB(i,m)..............................
!..............................D(i,j) for the pointed K-mesh..............................
D=0
   do ii=1,3
   do jj=1,3
D(ii,jj)=kt(1,ii,jj)
   enddo
   enddo
do ii=1,3
   do jj=4,6
     D(ii,jj)=kt(3,ii,mod(jj,3)+1)
   enddo
enddo
do ii=4,6
   do jj=1,3
     D(ii,jj)=kt(2,mod(ii,3)+1,jj)
   enddo
enddo
 do ii=4,6
   do jj=4,6
   D(ii,jj)=kt(4,MOD(ii,3)+1,mod(jj,3)+1)
   enddo
enddo
   write(11,*)"------------"
   !write(11,'(6f8.3,6f8.3)')D
   write(11,'(6f8.3)')real(D)
   write(*,*)"The D(i,j) at Kpoints:",kk_site(:,k)
   write(*,'(6f8.3)')real(D)
   !pause
call EVLCG(6,D,6,bzz)
do i=1,6
bzzR(k,i)=bzz(i)
enddo
write(22,'(i5,2x,6f12.6)')k,bzzR(k,:)
enddo 
!..............................D(i,j) for the pointed K-mesh..............................
!..............................K---space..............................
end
