"""
URL configuration for SplitNN_Program project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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

from neural_network import views as nn_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("predict/", nn_views.predict, name="predict"),
    path("train/", nn_views.train, name="test"),
    path("test/", nn_views.test, name="train"),
    # path("reset_runner/", nn_views.restart_runner, name="reset_runner"),
    path("report_client_nn_reset/", nn_views.report_client_nn_reset, name="report_client_nn_reset"),
    path("save_reports/", nn_views.save_reports, name="save_reports"),
    path("current_client/", nn_views.current_client, name="current_client"),
    path("prepare_running/", nn_views.prepare_running, name="prepare_running"),

]
