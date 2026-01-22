# -*- coding: utf-8 -*-
"""
远程文件上传服务
"""
import os
import logging
import paramiko
from typing import Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class RemoteUploader:
    """远程文件上传器"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def upload(self, local_file_path: str, remote_filename: str) -> Dict[str, Any]:
        """
        将本地文件通过SFTP上传到远程服务器
        
        Args:
            local_file_path: 本地文件路径
            remote_filename: 远程文件名
            
        Returns:
            上传结果字典
        """
        ssh_client = None
        sftp = None
        
        try:
            remote_file_path = os.path.join(
                self.settings.remote_base_path, 
                remote_filename
            ).replace("\\", "/")
            
            # 创建SSH连接
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                hostname=self.settings.remote_host,
                port=self.settings.remote_port,
                username=self.settings.remote_user,
                password=self.settings.remote_password
            )
            
            # SFTP上传
            sftp = ssh_client.open_sftp()
            sftp.put(local_file_path, remote_file_path, confirm=True)
            
            logger.info(f"文件上传成功: {remote_file_path}")
            
            return {
                "success": True,
                "message": f"文件已上传至远程服务器",
                "remote_path": remote_file_path
            }
            
        except Exception as e:
            logger.error(f"远程上传失败: {e}")
            return {
                "success": False,
                "message": f"远程上传失败: {str(e)}"
            }
            
        finally:
            if sftp:
                sftp.close()
            if ssh_client:
                ssh_client.close()
