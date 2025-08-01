"""Fairness_and_Composition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Law_case import views as mainView
from admins import views as admins
from users import views as usr
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings
from utility.Bailable import bailable,pred_bail
from utility.punishment import predict_punishment,punish

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", mainView.index, name="index"),
    path("index/", mainView.index, name="index"),
    path("Adminlogin/", mainView.AdminLogin, name="AdminLogin"),
    path("UserLogin/", mainView.UserLogin, name="UserLogin"),
    path('userhome',mainView.userhome,name='userhome'),
    path("UserRegister/", mainView.UserRegister, name="UserRegister"),

    # adminviews
    path("AdminLoginCheck/", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path('RegisterUsersView/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivaUsers/', admins.ActivaUsers, name='ActivaUsers'),
    path('DeleteUsers/', admins.DeleteUsers, name='DeleteUsers'),
    path('BlockUsers/', admins.BlockUsers, name='BlockUsers'),
    path("dataset", admins.data_v, name='dataset'),

    # User Views
    path("UserRegisterActions", usr.UserRegisterActions, name="UserRegisterActions"),
    path("UserLoginCheck", usr.UserLoginCheck, name="UserLoginCheck"),
    path("UserHome", usr.UserHome, name="UserHome"),


    path("bailable", bailable, name="bailable"),
    path('pred_bail', pred_bail, name='pred_bail'),
    path('training,',punish,name='training'),
    path('punish',predict_punishment,name='punish')
    
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
# urlpatterns += staticfiles_urlpatterns()
# urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
