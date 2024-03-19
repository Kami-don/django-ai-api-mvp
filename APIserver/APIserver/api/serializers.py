# from django.contrib.auth.models import Group, User
from rest_framework import serializers
from django.core.validators import URLValidator
from django.core.files.base import ContentFile
from django.core.files import File
from rest_framework.serializers import ValidationError
from rest_framework import serializers

from urllib.request import urlretrieve


# class UserSerializer(serializers.HyperlinkedModelSerializer):

#     creator = serializers.ReadOnlyField(source='creator.username')
#     class Meta:
#         model = User
#         fields = ['url', 'username', 'email', 'groups']

class FileUrlField(serializers.FileField):
    def to_internal_value(self, data):
        try:
            URLValidator()(data)
        except ValidationError as e:
            raise ValidationError('Invalid Url')

        # download the contents from the URL
        file, http_message = urlretrieve(data)
        file = File(open(file, 'rb'))
        return super(FileUrlField, self).to_internal_value(ContentFile(file.read(), name=file.name))