"""
URL routing for analytics application
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('upload/', views.upload_dataset_view, name='upload_dataset'),
    path('data-summary/', views.data_summary_view, name='data_summary'),
    path('dashboard/', views.visual_dashboard_view, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
    path('api/predict/', views.api_predict_view, name='api_predict'),
    path('lab-report/', views.lab_report_view, name='lab_report'),
]
